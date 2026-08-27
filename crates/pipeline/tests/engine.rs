//! Pipeline behaviour end to end.
//!
//! Engine logic — segmentation, draft/final ordering, timestamp arithmetic — is
//! tested with a stub recogniser so it runs anywhere. Real transcription needs
//! a 643 MB model, so those tests skip themselves when it is absent; run
//! `scripts/fetch-models.sh` to enable them.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use shi_audio::StreamKind;
use shi_pipeline::{
    ModelPaths, PipelineEvent, SherpaTranscriber, StreamPipeline, Transcriber, Transcript,
    VadSettings,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn silero() -> Option<String> {
    let path = repo_root().join("models/silero_vad.onnx");
    path.is_file().then(|| path.to_string_lossy().into_owned())
}

fn parakeet_dir() -> Option<PathBuf> {
    let dir = repo_root().join("models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8");
    dir.is_dir().then_some(dir)
}

/// Read the fixture as 16 kHz mono f32, matching what capture delivers.
fn fixture() -> Vec<f32> {
    let path = repo_root().join("fixtures/ru-two-turns.wav");
    let mut reader = hound::WavReader::open(&path).expect("open fixture");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "fixture must already be 16 kHz");
    assert_eq!(spec.channels, 1, "fixture must be mono");
    reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect()
}

/// Returns one predictable word per second of audio, so tests can assert on
/// structure without depending on a real model's output.
struct StubTranscriber {
    calls: AtomicUsize,
}

impl StubTranscriber {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Transcriber for StubTranscriber {
    fn transcribe(&self, samples: &[f32]) -> shi_pipeline::Result<Transcript> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let words = (samples.len() / 16_000).max(1);
        Ok(Transcript {
            text: vec!["слово"; words].join(" "),
            tokens: vec!["слово".into(); words],
            token_offsets: (0..words).map(|i| Duration::from_secs(i as u64)).collect(),
        })
    }
    fn model_id(&self) -> &str {
        "stub"
    }
}

/// Feed a fixture through a pipeline in realistic 20 ms blocks.
fn run(pipeline: &mut StreamPipeline, audio: &[f32]) -> Vec<PipelineEvent> {
    let mut events = Vec::new();
    for block in audio.chunks(320) {
        events.extend(pipeline.push(block));
    }
    events.extend(pipeline.flush());
    events
}

#[test]
fn a_pause_splits_speech_into_separate_utterances() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent — run scripts/fetch-models.sh");
        return;
    };

    let stub = Arc::new(StubTranscriber::new());
    let mut pipeline = StreamPipeline::new(
        StreamKind::System,
        16_000,
        &silero,
        stub.clone(),
        VadSettings::default(),
    )
    .expect("build pipeline");

    let events = run(&mut pipeline, &fixture());
    let finals: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            PipelineEvent::Final { start, end, .. } => Some((*start, *end)),
            _ => None,
        })
        .collect();

    assert_eq!(
        finals.len(),
        2,
        "the fixture holds two turns around a 900 ms pause, got {finals:?}"
    );

    let (first_start, first_end) = finals[0];
    let (second_start, second_end) = finals[1];
    assert!(first_end > first_start, "an utterance must have duration");
    assert!(
        second_start >= first_end,
        "utterances must not overlap: {first_end:?} then {second_start:?}"
    );
    assert!(
        second_end <= Duration::from_secs(5),
        "timestamps must stay within the 4.2 s fixture, got {second_end:?}"
    );
    assert!(stub.calls() > 0);
}

#[test]
fn every_stream_kind_is_carried_through() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    for kind in [StreamKind::Mic, StreamKind::System] {
        let mut pipeline = StreamPipeline::new(
            kind,
            16_000,
            &silero,
            Arc::new(StubTranscriber::new()),
            VadSettings::default(),
        )
        .expect("build pipeline");

        let events = run(&mut pipeline, &fixture());
        assert!(!events.is_empty(), "{kind} produced nothing");
        assert!(
            events.iter().all(|e| e.stream() == kind),
            "{kind} pipeline emitted events tagged with another stream"
        );
    }
}

#[test]
fn a_resampling_source_still_produces_utterances() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    // Capture devices run at 48 kHz, so this is the path real audio takes.
    let upsampled: Vec<f32> = fixture().iter().flat_map(|s| [*s, *s, *s]).collect();

    let mut pipeline = StreamPipeline::new(
        StreamKind::Mic,
        48_000,
        &silero,
        Arc::new(StubTranscriber::new()),
        VadSettings::default(),
    )
    .expect("build pipeline");

    let events = run(&mut pipeline, &upsampled);
    let finals = events
        .iter()
        .filter(|e| matches!(e, PipelineEvent::Final { .. }))
        .count();
    assert!(finals > 0, "48 kHz input produced no utterances");
}

#[test]
fn real_model_transcribes_the_fixture() {
    let (Some(silero), Some(model_dir)) = (silero(), parakeet_dir()) else {
        eprintln!("skipping: models absent — run scripts/fetch-models.sh");
        return;
    };

    let transcriber = Arc::new(
        SherpaTranscriber::load(&ModelPaths::parakeet_int8(&model_dir), 4).expect("load model"),
    );
    let mut pipeline = StreamPipeline::new(
        StreamKind::System,
        16_000,
        &silero,
        transcriber,
        VadSettings::default(),
    )
    .expect("build pipeline");

    let events = run(&mut pipeline, &fixture());
    let text: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            PipelineEvent::Final { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();

    assert!(!text.is_empty(), "no final transcript produced");
    let joined = text.join(" ").to_lowercase();
    for expected in ["релиз", "миграц"] {
        assert!(
            joined.contains(expected),
            "expected {expected:?} somewhere in {joined:?}"
        );
    }
}
