use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Counters shared between a realtime capture callback and the UI.
///
/// Every operation here is a relaxed atomic — no allocation, no locking, no
/// syscalls — so it is safe to call from an audio thread.
///
/// Two of these counters exist because of failures that are otherwise silent:
///
/// - `frames_dropped`: a stalled consumer leaves a gap in the transcript. We
///   want that visible while the meeting runs, not found later in the Markdown.
/// - `saw_signal`: macOS denies system-audio capture by handing back an endless
///   stream of zeroes rather than an error. "Stream started" therefore proves
///   nothing; only a non-zero sample does.
#[derive(Debug, Default)]
pub struct StreamStats {
    frames_captured: AtomicU64,
    frames_dropped: AtomicU64,
    /// Peak absolute sample since the last `take_peak`, as `f32::to_bits`.
    peak_bits: AtomicU32,
    /// Set once any sample above [`SILENCE_FLOOR`] has been seen.
    saw_signal: AtomicBool,
}

/// Below this a block is indistinguishable from digital silence. Chosen well
/// under any real room noise but above float denormal noise.
pub const SILENCE_FLOOR: f32 = 1e-6;

impl StreamStats {
    pub fn record_captured(&self, frames: u64) {
        self.frames_captured.fetch_add(frames, Ordering::Relaxed);
    }

    pub fn record_dropped(&self, frames: u64) {
        self.frames_dropped.fetch_add(frames, Ordering::Relaxed);
    }

    /// Keep the loudest sample seen since the meter last read us.
    pub fn record_peak(&self, peak: f32) {
        let incoming = peak.abs();
        if incoming > SILENCE_FLOOR && !self.saw_signal.load(Ordering::Relaxed) {
            self.saw_signal.store(true, Ordering::Relaxed);
        }
        let mut current = self.peak_bits.load(Ordering::Relaxed);
        loop {
            if f32::from_bits(current) >= incoming {
                return;
            }
            match self.peak_bits.compare_exchange_weak(
                current,
                incoming.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn frames_captured(&self) -> u64 {
        self.frames_captured.load(Ordering::Relaxed)
    }

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    /// Whether this stream has ever produced a non-silent sample.
    ///
    /// A stream that is running, accumulating frames, and has never set this is
    /// the signature of a capture permission that was refused rather than
    /// reported.
    pub fn has_signal(&self) -> bool {
        self.saw_signal.load(Ordering::Relaxed)
    }

    /// Read the peak level and reset it, so the meter decays when audio stops.
    pub fn take_peak(&self) -> f32 {
        f32::from_bits(self.peak_bits.swap(0, Ordering::Relaxed))
    }
}
