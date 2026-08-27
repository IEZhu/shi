use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::error::{AudioError, Result};
use crate::ring::RingWriter;
use crate::source::{AudioSource, SourceInfo, StreamHandle, StreamKind, ring_capacity};
use crate::stats::StreamStats;

/// How quickly a [`FileSource`] replays its fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pacing {
    /// Match wall clock, as a real device would. Use when eyeballing the UI.
    Realtime,
    /// Push as fast as the consumer drains, waiting rather than dropping when
    /// the ring is full. Deterministic and lossless — this is what CI uses.
    Immediate,
}

/// Replays a WAV file as if it were a capture device.
///
/// This is not a convenience: it is the mechanism that keeps the pipeline
/// portable. Every test above this layer runs against a `FileSource`, so if a
/// test ever needs real hardware, a platform assumption has leaked into code
/// that should not have one.
pub struct FileSource {
    path: PathBuf,
    kind: StreamKind,
    pacing: Pacing,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl FileSource {
    pub fn new(path: impl AsRef<Path>, kind: StreamKind, pacing: Pacing) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            kind,
            pacing,
            running: Arc::new(AtomicBool::new(false)),
            worker: None,
        }
    }

    /// Read a WAV into interleaved f32, normalising whatever the file holds.
    fn read_interleaved(path: &Path) -> Result<(Vec<f32>, hound::WavSpec)> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| AudioError::Device(format!("cannot open {}: {e}", path.display())))?;
        let spec = reader.spec();

        let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, _) => reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| AudioError::Device(format!("bad float samples: {e}")))?,
            (hound::SampleFormat::Int, bits) => {
                // i32 covers every integer width hound produces; scale by the
                // width actually stored so 16- and 24-bit files match levels.
                let scale = 1.0 / (1i64 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| AudioError::Device(format!("bad int samples: {e}")))?
            }
        };

        Ok((samples, spec))
    }
}

impl AudioSource for FileSource {
    fn start(&mut self) -> Result<StreamHandle> {
        if self.worker.is_some() {
            return Err(AudioError::AlreadyRunning);
        }

        let (samples, spec) = Self::read_interleaved(&self.path)?;
        let channels = spec.channels;
        let sample_rate = spec.sample_rate;

        let info = SourceInfo {
            kind: self.kind,
            device_name: format!("file:{}", self.path.display()),
            sample_rate,
            channels,
        };

        let stats = Arc::new(StreamStats::default());
        let (producer, consumer) =
            rtrb::RingBuffer::new(ring_capacity(sample_rate, crate::source::DEFAULT_RING_SECONDS));
        let mut writer = RingWriter::new(producer, Arc::clone(&stats));

        // ~32 ms at 16 kHz, close to a typical device callback size.
        const BLOCK_FRAMES: usize = 512;
        let block_samples = BLOCK_FRAMES * channels.max(1) as usize;
        let block_duration =
            Duration::from_secs_f64(BLOCK_FRAMES as f64 / sample_rate.max(1) as f64);

        self.running.store(true, Ordering::SeqCst);
        let running = Arc::clone(&self.running);
        let pacing = self.pacing;

        self.worker = Some(thread::spawn(move || {
            for block in samples.chunks(block_samples) {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                match pacing {
                    Pacing::Realtime => {
                        writer.write_interleaved(block, channels);
                        thread::sleep(block_duration);
                    }
                    Pacing::Immediate => {
                        // Wait for room *before* writing rather than retrying
                        // after a partial write, which would duplicate the
                        // frames that already made it in. A fixture replay must
                        // be lossless or test assertions mean nothing.
                        let needed = block.len() / channels.max(1) as usize;
                        while writer.slots() < needed {
                            if !running.load(Ordering::SeqCst) {
                                return;
                            }
                            thread::sleep(Duration::from_micros(200));
                        }
                        writer.write_interleaved(block, channels);
                    }
                }
            }
            running.store(false, Ordering::SeqCst);
        }));

        Ok(StreamHandle {
            info,
            consumer,
            stats,
        })
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    fn kind(&self) -> StreamKind {
        self.kind
    }
}

impl Drop for FileSource {
    fn drop(&mut self) {
        self.stop();
    }
}
