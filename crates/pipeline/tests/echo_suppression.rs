//! Echo suppression through the real pipeline.
//!
//! The acoustics of a room are not reproducible, so the loudspeaker path is
//! simulated instead: the same audio, delayed, attenuated and dulled, exactly
//! as a microphone would receive it. Everything else — resampling, the real
//! voice-activity detector, utterance boundaries — is the production path.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use shi_audio::StreamKind;
use shi_pipeline::{
    EchoReference, PipelineEvent, StreamPipeline, Transcriber, Transcript, VadSettings,
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

fn fixture() -> Vec<f32> {
    let path = repo_root().join("fixtures/ru-two-turns.wav");
    let mut reader = hound::WavReader::open(&path).expect("open fixture");
    reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect()
}

/// What a loudspeaker and a room do before the microphone hears it.
fn through_the_air(source: &[f32], delay: Duration, gain: f32) -> Vec<f32> {
    let delay_samples = (delay.as_secs_f32() * 16_000.0) as usize;
    let mut low_passed = 0.0f32;
    let mut out = vec![0.0f32; delay_samples];
    for sample in source {
        low_passed = low_passed * 0.7 + sample * 0.3;
        out.push(low_passed * gain);
    }
    out
}

struct AnyText;
impl Transcriber for AnyText {
    fn transcribe(&self, samples: &[f32]) -> shi_pipeline::Result<Transcript> {
        Ok(Transcript {
            text: format!("{} сэмплов", samples.len()),
            tokens: vec!["сэмплов".into()],
            token_offsets: vec![Duration::ZERO],
        })
    }
    fn model_id(&self) -> &str {
        "any-text"
    }
}

fn build(kind: StreamKind, silero: &str) -> StreamPipeline {
    StreamPipeline::new(
        kind,
        16_000,
        silero,
        Arc::new(AnyText),
        VadSettings::default(),
    )
    .expect("build pipeline")
}

fn drive(pipeline: &mut StreamPipeline, audio: &[f32]) -> Vec<PipelineEvent> {
    let mut events = Vec::new();
    for block in audio.chunks(320) {
        events.extend(pipeline.push(block));
    }
    events.extend(pipeline.flush());
    events
}

/// Feed audio to a source that delivered nothing for `gap` first.
///
/// `push` reads the wall clock itself, so a test that pushes as fast as it can
/// never reproduces a quiet source. Stating the clock is the only way to make
/// the pipeline insert the silence it inserts in a real meeting.
fn drive_after_gap(
    pipeline: &mut StreamPipeline,
    audio: &[f32],
    gap: Duration,
) -> Vec<PipelineEvent> {
    let mut events = Vec::new();
    let mut elapsed = gap;
    for block in audio.chunks(320) {
        elapsed += Duration::from_secs_f32(block.len() as f32 / 16_000.0);
        events.extend(pipeline.push_at(block, elapsed));
    }
    events.extend(pipeline.flush());
    events
}

fn finals(events: &[PipelineEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, PipelineEvent::Final { .. }))
        .count()
}

#[test]
fn the_microphone_hearing_the_speakers_is_discarded() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent — run scripts/fetch-models.sh");
        return;
    };

    let remote = fixture();
    let reference = Arc::new(EchoReference::default());

    // The system stream publishes what it played.
    let mut system = build(StreamKind::System, &silero);
    system.publish_echo_reference(Arc::clone(&reference));
    let system_events = drive(&mut system, &remote);
    assert!(finals(&system_events) > 0, "the reference stream said nothing");

    // The microphone hears the very same audio through the air.
    let mut mic = build(StreamKind::Mic, &silero);
    mic.suppress_echo_of(Arc::clone(&reference));
    let mic_events = drive(&mut mic, &through_the_air(&remote, Duration::from_millis(120), 0.35));

    assert_eq!(
        finals(&mic_events),
        0,
        "echoed utterances reached the transcript: {mic_events:#?}"
    );
    assert!(
        mic.suppressed_echo() > 0,
        "suppression must be counted so a misfire is visible to the user"
    );
}

#[test]
fn the_local_speaker_survives_suppression() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    // Nothing is playing through the speakers, so nothing should be dropped.
    let reference = Arc::new(EchoReference::default());
    let mut mic = build(StreamKind::Mic, &silero);
    mic.suppress_echo_of(Arc::clone(&reference));

    let events = drive(&mut mic, &fixture());
    assert!(
        finals(&events) > 0,
        "suppression swallowed the user's own speech"
    );
    assert_eq!(mic.suppressed_echo(), 0);
}

#[test]
fn talking_over_the_remote_side_is_not_suppressed() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    // The hardest case: both are speaking, so the reference is full of audio
    // that simply is not what the microphone is picking up.
    let remote = fixture();
    let reference = Arc::new(EchoReference::default());
    let mut system = build(StreamKind::System, &silero);
    system.publish_echo_reference(Arc::clone(&reference));
    drive(&mut system, &remote);

    // The user says something of their own at the same time: the fixture
    // played backwards has the same spectrum but a different contour.
    let mut local: Vec<f32> = remote.clone();
    local.reverse();

    let mut mic = build(StreamKind::Mic, &silero);
    mic.suppress_echo_of(Arc::clone(&reference));
    let events = drive(&mut mic, &local);

    assert!(
        finals(&events) > 0,
        "the user was silenced while the other side talked; suppressed {}",
        mic.suppressed_echo()
    );
}

#[test]
fn suppression_survives_a_quiet_stretch_before_anyone_speaks() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    // A meeting begins, and for a while nobody says anything. Both streams
    // insert silence to keep their clocks honest — and the reference has to
    // carry that silence too, or every later utterance is compared against
    // audio from a different moment.
    let gap = Duration::from_secs(3);
    let remote = fixture();

    let reference = Arc::new(EchoReference::default());
    let mut system = build(StreamKind::System, &silero);
    system.publish_echo_reference(Arc::clone(&reference));
    let system_events = drive_after_gap(&mut system, &remote, gap);
    assert!(finals(&system_events) > 0, "the reference stream said nothing");

    let mut mic = build(StreamKind::Mic, &silero);
    mic.suppress_echo_of(Arc::clone(&reference));
    let heard = through_the_air(&remote, Duration::from_millis(120), 0.35);
    let mic_events = drive_after_gap(&mut mic, &heard, gap);

    assert_eq!(
        finals(&mic_events),
        0,
        "a quiet start slid the reference off the microphone's clock, so the \
         speakers were transcribed a second time: {mic_events:#?}"
    );
    assert!(mic.suppressed_echo() > 0);
}

#[test]
fn speaking_over_the_echo_is_not_suppressed() {
    let Some(silero) = silero() else {
        eprintln!("skipping: models/silero_vad.onnx absent");
        return;
    };

    // Double talk, the case that decides whether suppression is safe to ship:
    // the microphone carries the user's own voice *and* the speakers echoing
    // the remote side at the same time. Dropping that utterance would delete
    // something only the user said, which no amount of echo removal is worth.
    let remote = fixture();
    let reference = Arc::new(EchoReference::default());
    let mut system = build(StreamKind::System, &silero);
    system.publish_echo_reference(Arc::clone(&reference));
    drive(&mut system, &remote);

    let echo = through_the_air(&remote, Duration::from_millis(120), 0.35);
    let mut local: Vec<f32> = remote.clone();
    local.reverse();

    let mixed: Vec<f32> = echo
        .iter()
        .zip(local.iter())
        .map(|(e, l)| (e + l).clamp(-1.0, 1.0))
        .collect();

    let mut mic = build(StreamKind::Mic, &silero);
    mic.suppress_echo_of(Arc::clone(&reference));
    let events = drive(&mut mic, &mixed);

    assert!(
        finals(&events) > 0,
        "the user was silenced for talking at the same time as the speakers; \
         suppressed {}",
        mic.suppressed_echo()
    );
}
