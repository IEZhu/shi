//! The voice gate through the real pipeline.
//!
//! `AnyText` stands in for the recogniser and returns a line for whatever it is
//! given, which is exactly the behaviour being defended against: over a real
//! meeting a willing recogniser turned breath into "Thank you." thirteen times.
//! A stub that never returns nothing makes the gate the only thing standing
//! between noise and the transcript, so these tests fail loudly if it stops
//! working.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use shi_audio::StreamKind;
use shi_pipeline::{
    PipelineEvent, SpeakerTracker, StreamPipeline, Thresholds, Transcriber, Transcript,
    VadSettings, holds_speech, voiced_run_ms,
};

const RATE: u32 = 16_000;

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

/// Four seconds of the room a real meeting was held in.
///
/// Cut from the microphone stream of that recording at the point where the
/// recogniser announced "Yeah." over an empty room. Synthetic noise will not do
/// here: white noise opens no segment at all, and shaping it into something
/// Silero accepts means tuning a signal until it proves the point, which proves
/// nothing. This is the audio that actually caused the bug.
fn room_tone() -> Vec<f32> {
    let path = repo_root().join("fixtures/room-tone-mic.wav");
    let mut reader = hound::WavReader::open(&path).expect("open room tone fixture");
    reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect()
}

struct AnyText;
impl Transcriber for AnyText {
    fn transcribe(&self, _: &[f32]) -> shi_pipeline::Result<Transcript> {
        Ok(Transcript {
            text: "Thank you.".into(),
            tokens: vec!["Thank".into(), "you.".into()],
            token_offsets: vec![Duration::ZERO, Duration::ZERO],
        })
    }
    fn model_id(&self) -> &str {
        "any-text"
    }
}

fn build(silero: &str) -> StreamPipeline {
    build_with(silero, VadSettings::default())
}

fn build_with(silero: &str, settings: VadSettings) -> StreamPipeline {
    StreamPipeline::new(StreamKind::Mic, RATE, silero, Arc::new(AnyText), settings)
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

fn finals(events: &[PipelineEvent]) -> usize {
    events.iter().filter(|e| matches!(e, PipelineEvent::Final { .. })).count()
}

#[test]
fn the_room_that_produced_a_thank_you_is_rejected() {
    // The regression, stated as directly as it can be: this exact audio was
    // handed to a recogniser during a real meeting and came back as "Yeah."
    assert!(!holds_speech(&room_tone(), RATE), "an empty room is not speech");
    assert!(holds_speech(&fixture(), RATE), "and a person is");
}

#[test]
fn the_detector_only_opens_on_that_room_mid_meeting() {
    // Worth pinning because it explains why no fixture ever caught this. Fed
    // the same four seconds on their own, Silero opens nothing at all; it
    // opened a segment there only with an hour of meeting behind it. Any test
    // that tries to reproduce the original failure from a short clip is
    // testing its own signal, not the pipeline.
    let Some(silero) = silero() else { return };
    let mut pipeline = build(&silero);

    let events = drive(&mut pipeline, &room_tone());

    assert_eq!(finals(&events), 0, "nothing was said, so nothing may be printed");
    assert_eq!(
        pipeline.voiceless_segments(),
        0,
        "the detector declined it first, so the gate never had to"
    );
}

#[test]
fn speech_still_gets_through() {
    let Some(silero) = silero() else { return };
    let mut pipeline = build(&silero);

    let events = drive(&mut pipeline, &fixture());

    assert!(finals(&events) > 0, "real speech must survive the gate");
    assert_eq!(pipeline.voiceless_segments(), 0, "and must not be counted against it");
}

#[test]
fn a_faint_talker_is_not_mistaken_for_a_quiet_room() {
    // The failure this gate exists to avoid repeating. Loudness was the
    // obvious test and would have silenced this person; the measurement that
    // rejected it is in docs/transcription.md.
    let Some(silero) = silero() else { return };
    let faint: Vec<f32> = fixture().iter().map(|s| s * 0.05).collect();

    let mut pipeline = build(&silero);
    let events = drive(&mut pipeline, &faint);

    assert!(finals(&events) > 0, "a quiet voice is still a voice");
    assert_eq!(pipeline.voiceless_segments(), 0);
}

#[test]
fn the_score_does_not_move_when_the_level_does() {
    let loud = fixture();
    let faint: Vec<f32> = loud.iter().map(|s| s * 0.02).collect();

    let a = voiced_run_ms(&loud, RATE, u32::MAX);
    let b = voiced_run_ms(&faint, RATE, u32::MAX);
    assert_eq!(a, b, "loud {a} ms, faint {b} ms");
}

#[test]
fn the_recogniser_is_never_handed_more_than_it_can_take() {
    // The detector is asked to close utterances at `max_speech` and on real
    // recordings ran to one and a half times that — and, with another detector
    // model, to over two minutes, which the encoder answered with an exception
    // that ended the process. Whatever the detector does, no final may exceed
    // the pipeline's own cap, and the speech inside still has to come out.
    let Some(silero) = silero() else { return };
    let settings = VadSettings {
        longest_decode: Duration::from_millis(1_500),
        min_voiced_ms: 0,
        ..VadSettings::default()
    };
    let cap = settings.longest_decode + Duration::from_millis(30);

    let mut pipeline = build_with(&silero, settings);
    let events = drive(&mut pipeline, &fixture());

    let lengths: Vec<Duration> = events
        .iter()
        .filter_map(|e| match e {
            PipelineEvent::Final { start, end, .. } => Some(end.saturating_sub(*start)),
            _ => None,
        })
        .collect();
    assert!(!lengths.is_empty(), "the speech must still be transcribed");
    assert!(
        lengths.iter().all(|l| *l <= cap),
        "a final exceeded the cap: {lengths:?}"
    );
    assert!(lengths.len() >= 2, "a four-second fixture must arrive in pieces at this cap");
}

#[test]
fn the_pieces_of_one_utterance_belong_to_one_speaker() {
    // A decode is more accurate the shorter it is and an embedding the longer,
    // so the recogniser gets five-second pieces while the voice is judged from
    // the whole utterance. Judging each piece instead turned one real meeting
    // into a hundred and forty-four speakers.
    let Some(silero) = silero() else { return };
    let model = repo_root().join("models/nemo_en_titanet_small.onnx");
    if !model.is_file() {
        return;
    }
    let Ok(tracker) = SpeakerTracker::new(&model.to_string_lossy(), 2, Thresholds::default())
    else {
        return;
    };

    // A cap far below the fixture's length forces the split.
    let settings = VadSettings {
        longest_decode: Duration::from_millis(700),
        min_voiced_ms: 0,
        ..VadSettings::default()
    };
    let mut pipeline =
        StreamPipeline::new(StreamKind::System, RATE, &silero, Arc::new(AnyText), settings)
            .expect("build pipeline");
    pipeline.identify_speakers(tracker);

    let events = drive(&mut pipeline, &fixture());
    let speakers: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            PipelineEvent::Final { speaker, .. } => Some(format!("{speaker:?}")),
            _ => None,
        })
        .collect();

    assert!(speakers.len() >= 2, "the fixture must arrive in pieces at this cap");
    let distinct: std::collections::BTreeSet<&String> = speakers.iter().collect();
    assert!(
        distinct.len() <= 2,
        "one utterance was split across {} verdicts: {speakers:?}",
        distinct.len()
    );
}
