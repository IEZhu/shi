use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::Serialize;
use shi_audio::{AudioSource, MicSource, StreamHandle, StreamKind, SystemSource};
use tauri::{AppHandle, Emitter};

/// How often the readiness meters refresh. Fast enough to look live, slow
/// enough that the UI thread is not the bottleneck.
const TICK: Duration = Duration::from_millis(50);

/// Event name the frontend subscribes to.
pub const READINESS_EVENT: &str = "readiness";

/// What the readiness panel needs to know about one stream.
///
/// `verdict` is deliberately separate from `running`: on macOS a refused
/// system-audio grant produces a stream that is running, accumulating frames,
/// and completely silent. Only [`Verdict::Silent`] catches that.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamStatus {
    pub kind: String,
    pub verdict: Verdict,
    pub device_name: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub frames_captured: u64,
    pub frames_dropped: u64,
    /// Peak level since the previous tick, 0.0..=1.0.
    pub peak: f32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Verdict {
    /// Not started.
    Stopped,
    /// The source refused to start; `error` explains why.
    Failed,
    /// Running but no frames have arrived at all.
    NoFrames,
    /// Frames arrive and every one of them is zero. On macOS this is what a
    /// denied capture permission looks like — see docs/capture-macos.md.
    Silent,
    /// Frames arrive and carry real audio.
    Ok,
}

impl StreamStatus {
    fn stopped(kind: StreamKind) -> Self {
        Self {
            kind: kind.as_str().to_string(),
            verdict: Verdict::Stopped,
            device_name: None,
            sample_rate: None,
            channels: None,
            frames_captured: 0,
            frames_dropped: 0,
            peak: 0.0,
            error: None,
        }
    }

    fn failed(kind: StreamKind, error: String) -> Self {
        Self {
            verdict: Verdict::Failed,
            error: Some(error),
            ..Self::stopped(kind)
        }
    }

    fn sample(handle: &StreamHandle) -> Self {
        let captured = handle.stats.frames_captured();
        let verdict = if captured == 0 {
            Verdict::NoFrames
        } else if !handle.stats.has_signal() {
            Verdict::Silent
        } else {
            Verdict::Ok
        };

        Self {
            kind: handle.info.kind.as_str().to_string(),
            verdict,
            device_name: Some(handle.info.device_name.clone()),
            sample_rate: Some(handle.info.sample_rate),
            channels: Some(handle.info.channels),
            frames_captured: captured,
            frames_dropped: handle.stats.frames_dropped(),
            peak: handle.stats.take_peak(),
            error: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub mic: StreamStatus,
    pub system: StreamStatus,
}

impl Readiness {
    fn idle() -> Self {
        Self {
            mic: StreamStatus::stopped(StreamKind::Mic),
            system: StreamStatus::stopped(StreamKind::System),
        }
    }
}

/// Owns both capture sources and the thread that meters them.
pub struct Capture {
    mic: MicSource,
    system: SystemSource,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    /// Shared with the metering thread so a `readiness` query is always current,
    /// even if the frontend missed an event or reconnected late.
    last: Arc<Mutex<Readiness>>,
}

impl Capture {
    pub fn new() -> Self {
        Self {
            mic: MicSource::default_device(),
            system: SystemSource::new(),
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            last: Arc::new(Mutex::new(Readiness::idle())),
        }
    }

    pub fn snapshot(&self) -> Readiness {
        self.last
            .lock()
            .map(|r| r.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    /// Start both streams. Either may fail independently — a working
    /// microphone with a broken tap is still worth reporting honestly.
    pub fn start(&mut self, app: AppHandle) -> Readiness {
        if self.worker.is_some() {
            return self.snapshot();
        }

        let mic = self
            .mic
            .start()
            .map_err(|e| StreamStatus::failed(StreamKind::Mic, e.to_string()));
        let system = self
            .system
            .start()
            .map_err(|e| StreamStatus::failed(StreamKind::System, e.to_string()));

        let initial = Readiness {
            mic: mic.as_ref().map(StreamStatus::sample).unwrap_or_else(|e| e.clone()),
            system: system
                .as_ref()
                .map(StreamStatus::sample)
                .unwrap_or_else(|e| e.clone()),
        };
        self.store(initial.clone());

        if mic.is_err() && system.is_err() {
            return initial;
        }

        self.stop.store(false, Ordering::SeqCst);
        let stop = Arc::clone(&self.stop);
        let last = Arc::clone(&self.last);

        self.worker = Some(thread::spawn(move || {
            let mut mic = mic.ok();
            let mut system = system.ok();
            let mut tick: u32 = 0;

            while !stop.load(Ordering::SeqCst) {
                thread::sleep(TICK);

                // M1 replaces these drains with the transcription pipeline.
                // Until something consumes the rings they fill up and the drop
                // counter climbs, which would make the meters lie.
                let status = |handle: &mut Option<StreamHandle>, kind: StreamKind| match handle {
                    Some(h) => {
                        while h.consumer.pop().is_ok() {}
                        StreamStatus::sample(h)
                    }
                    None => StreamStatus::stopped(kind),
                };

                let readiness = Readiness {
                    mic: status(&mut mic, StreamKind::Mic),
                    system: status(&mut system, StreamKind::System),
                };

                // Once a second, leave a diagnosable trace. Capture problems
                // are reported by users as "it recorded nothing", and this is
                // what turns that into an answer.
                tick += 1;
                if tick % (1000 / TICK.as_millis() as u32).max(1) == 0 {
                    tracing::info!(
                        mic = ?readiness.mic.verdict,
                        mic_frames = readiness.mic.frames_captured,
                        mic_dropped = readiness.mic.frames_dropped,
                        system = ?readiness.system.verdict,
                        system_frames = readiness.system.frames_captured,
                        system_dropped = readiness.system.frames_dropped,
                        "readiness"
                    );
                }

                if let Ok(mut slot) = last.lock() {
                    *slot = readiness.clone();
                }
                if app.emit(READINESS_EVENT, &readiness).is_err() {
                    break;
                }
            }
        }));

        initial
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.mic.stop();
        self.system.stop();
        self.store(Readiness::idle());
    }

    fn store(&self, readiness: Readiness) {
        match self.last.lock() {
            Ok(mut slot) => *slot = readiness,
            Err(poisoned) => *poisoned.into_inner() = readiness,
        }
    }
}

impl Default for Capture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop();
    }
}
