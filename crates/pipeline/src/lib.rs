//! Turning captured audio into transcript segments.
//!
//! The pipeline runs once per capture stream. Both instances resample, detect
//! speech and transcribe identically; what differs is what happens to the
//! result — microphone audio belongs to one known speaker, while system audio
//! goes on to diarization.

pub mod asr;
pub mod cadence;
pub mod engine;
pub mod error;
pub mod event;

pub use asr::{ASR_SAMPLE_RATE, ModelPaths, SherpaTranscriber, Transcriber, Transcript};
pub use cadence::Cadence;
pub use engine::{StreamPipeline, VadSettings};
pub use event::PipelineEvent;
pub use error::{PipelineError, Result};
