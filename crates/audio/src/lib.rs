//! Capture for the meeting transcriber: microphone and system output as two
//! independent streams.
//!
//! Keeping them separate is the central design choice of the whole app. The
//! microphone carries one known speaker; the system output carries every
//! remote participant already mixed by the conferencing app. Only the second
//! needs diarization, which removes half the hard problem before any model
//! runs.
//!
//! Samples reach the ring at the device's native rate; conversion to
//! [`source::TARGET_SAMPLE_RATE`] happens on the consumer side so that capture
//! callbacks stay allocation-free.

pub mod error;
pub mod file;
pub mod mic;
pub mod ring;
pub mod source;
pub mod stats;
pub mod system;

pub use error::{AudioError, Result};
pub use file::{FileSource, Pacing};
pub use mic::MicSource;
pub use source::{
    AudioSource, DEFAULT_RING_SECONDS, SourceInfo, StreamHandle, StreamKind, TARGET_SAMPLE_RATE,
};
pub use stats::StreamStats;
pub use system::SystemSource;
