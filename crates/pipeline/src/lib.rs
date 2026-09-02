//! Turning captured audio into transcript segments.
//!
//! The pipeline runs once per capture stream. Both instances resample, detect
//! speech and transcribe identically; what differs is what happens to the
//! result — microphone audio belongs to one known speaker, while system audio
//! goes on to diarization.

pub mod asr;
pub mod cadence;
pub mod diarize;
pub mod echo;
pub mod preprocess;
pub mod speech;
pub mod engine;
pub mod error;
pub mod event;

pub use asr::{
    ASR_SAMPLE_RATE, ModelPaths, SherpaTranscriber, StreamingTranscriber, Transcriber,
    Transcript, load_recognizer,
};
pub use cadence::Cadence;
pub use diarize::{
    Attribution, LiveNames, Span, centroids, cluster, rediarize, DEFAULT_KNOWN_THRESHOLD, DEFAULT_SESSION_THRESHOLD, SessionSlot,
    SpeakerTracker, Thresholds, VoiceProfile,
};
pub use echo::{DEFAULT_ECHO_THRESHOLD, EchoReference};
pub use preprocess::{Denoiser, Settings as PreprocessSettings};
pub use engine::{AudioTap, StreamPipeline, VadSettings};
pub use speech::{MIN_VOICED_MS, SHORTEST_JUDGED, holds_speech, holds_speech_beyond, voiced_run_ms};
pub use event::PipelineEvent;
pub use error::{PipelineError, Result};
