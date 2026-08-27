use std::sync::Arc;

use crate::error::Result;
use crate::stats::StreamStats;

/// Everything downstream of capture works at 16 kHz mono: it is what the ASR,
/// VAD, segmentation and embedding models all expect.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// How much audio a ring buffer holds before a stalled consumer starts losing
/// samples. Generous on purpose — at 48 kHz this is ~5.8 MB, and the cost of
/// being wrong is a hole in a meeting transcript.
pub const DEFAULT_RING_SECONDS: f32 = 30.0;

/// Where a stream of audio physically came from.
///
/// This is the single most load-bearing distinction in the app. A `Mic` stream
/// carries one known speaker, so it needs verification but not diarization. A
/// `System` stream carries every remote participant mixed together by the
/// conferencing app, so it needs the full diarization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamKind {
    Mic,
    System,
}

impl StreamKind {
    /// Stable identifier, also used as the `segments.stream` column value.
    pub fn as_str(self) -> &'static str {
        match self {
            StreamKind::Mic => "mic",
            StreamKind::System => "system",
        }
    }
}

impl std::fmt::Display for StreamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a source is actually delivering, known only once capture has started.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub kind: StreamKind,
    pub device_name: String,
    /// Native rate of the hardware. Samples arrive at this rate; conversion to
    /// [`TARGET_SAMPLE_RATE`] happens on the consumer side.
    pub sample_rate: u32,
    /// Native channel count, before the callback's downmix to mono.
    pub channels: u16,
}

/// A running capture stream: the read end of its ring plus its live counters.
pub struct StreamHandle {
    pub info: SourceInfo,
    pub consumer: rtrb::Consumer<f32>,
    pub stats: Arc<StreamStats>,
}

/// A platform-specific way of getting audio into the pipeline.
///
/// The rest of the application is written against this trait and never learns
/// which OS it is on. Porting to Windows or Linux means adding an
/// implementation here, not touching anything downstream — and
/// [`FileSource`](crate::file::FileSource) means the whole pipeline can be
/// tested in CI with no audio hardware at all.
pub trait AudioSource: Send {
    /// Begin capturing. Returns the consumer end of a freshly created ring.
    fn start(&mut self) -> Result<StreamHandle>;

    /// Stop capturing and release the device. Idempotent.
    fn stop(&mut self);

    fn kind(&self) -> StreamKind;
}

/// Ring capacity in samples for a given rate, rounded up to whole seconds.
pub fn ring_capacity(sample_rate: u32, seconds: f32) -> usize {
    ((sample_rate as f32) * seconds).ceil() as usize
}
