use std::sync::Arc;

use rtrb::Producer;

use crate::stats::StreamStats;

/// The write end of a capture ring, used from inside realtime audio callbacks.
///
/// Hard rule for everything in this type: no allocation, no locking, no
/// syscalls, no logging. A callback that blocks drops samples, and dropped
/// samples are holes in the transcript.
pub struct RingWriter {
    producer: Producer<f32>,
    stats: Arc<StreamStats>,
}

impl RingWriter {
    pub fn new(producer: Producer<f32>, stats: Arc<StreamStats>) -> Self {
        Self { producer, stats }
    }

    /// Frames the ring can accept right now. Realtime-safe.
    pub fn slots(&self) -> usize {
        self.producer.slots()
    }

    /// Downmix an interleaved block to mono and push it into the ring.
    ///
    /// If the consumer has fallen behind, this writes what fits and counts the
    /// rest as dropped rather than blocking — a late transcript is recoverable,
    /// a stalled audio thread is not.
    pub fn write_interleaved(&mut self, data: &[f32], channels: u16) {
        let channels = channels.max(1) as usize;
        let frames = data.len() / channels;
        if frames == 0 {
            return;
        }

        // Peak is taken pre-downmix: the meter should show what the device is
        // actually receiving, not an average that hides a hot single channel.
        let mut peak = 0.0f32;
        for sample in data {
            peak = peak.max(sample.abs());
        }
        self.stats.record_peak(peak);

        let scale = 1.0 / channels as f32;
        let mono = data
            .chunks_exact(channels)
            .map(move |frame| frame.iter().sum::<f32>() * scale);

        let written = match self.producer.write_chunk_uninit(frames) {
            Ok(chunk) => chunk.fill_from_iter(mono),
            Err(rtrb::chunks::ChunkError::TooFewSlots(available)) => {
                match self.producer.write_chunk_uninit(available) {
                    Ok(chunk) => chunk.fill_from_iter(mono),
                    Err(_) => 0,
                }
            }
        };

        self.stats.record_captured(written as u64);
        if written < frames {
            self.stats.record_dropped((frames - written) as u64);
        }
    }
}
