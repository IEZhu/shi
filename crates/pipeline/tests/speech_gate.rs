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
    PipelineEvent, StreamPipeline, Transcriber, Transcript, VadSettings, holds_speech,
    voiced_run_ms,
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
    StreamPipeline::new(StreamKind::Mic, RATE, silero, Arc::new(AnyText), VadSettings::default())
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
