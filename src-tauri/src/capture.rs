use serde::Serialize;
use shi_audio::{AudioSource, MicSource, StreamHandle, StreamKind, SystemSource};

/// Event name carrying [`Readiness`] to the frontend.
pub const READINESS_EVENT: &str = "readiness";

/// What the readiness panel needs to know about one stream.
///
/// `verdict` is deliberately more than a boolean: on macOS a refused
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Verdict {
    Stopped,
    /// The source refused to start; `error` explains why.
    Failed,
    /// Running but no frames have arrived at all.
    NoFrames,
    /// Frames arrive and every one of them is zero. On macOS this is what a
    /// denied capture permission looks like — see docs/capture-macos.md.
    Silent,
    Ok,
}

impl StreamStatus {
    pub fn stopped(kind: StreamKind) -> Self {
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

    pub fn failed(kind: StreamKind, error: String) -> Self {
        Self {
            verdict: Verdict::Failed,
            error: Some(error),
            ..Self::stopped(kind)
        }
    }

    /// Sample a running stream. Consumes the peak, so only the metering thread
    /// should call this.
    pub fn sample(info: &shi_audio::SourceInfo, stats: &shi_audio::StreamStats) -> Self {
        let captured = stats.frames_captured();
        let verdict = if captured == 0 {
            Verdict::NoFrames
        } else if !stats.has_signal() {
            Verdict::Silent
        } else {
            Verdict::Ok
        };

        Self {
            kind: info.kind.as_str().to_string(),
            verdict,
            device_name: Some(info.device_name.clone()),
            sample_rate: Some(info.sample_rate),
            channels: Some(info.channels),
            frames_captured: captured,
            frames_dropped: stats.frames_dropped(),
            peak: stats.take_peak(),
            error: None,
        }
    }
}

/// Capture health plus anything else that must be true before a meeting.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub mic: StreamStatus,
    pub system: StreamStatus,
    /// Models the pipeline needs but cannot find. A missing model at the start
    /// of a call is the same class of failure as a missing permission.
    pub missing_models: Vec<String>,
    /// True while a meeting is being transcribed.
    pub recording: bool,
    /// Measured decode cost per second of audio, once transcription has run.
    pub rtf: Option<f32>,
    /// Utterances dropped because the microphone was hearing the speakers.
    pub echo_suppressed: u64,
}

impl Readiness {
    pub fn idle(missing_models: Vec<String>) -> Self {
        Self {
            mic: StreamStatus::stopped(StreamKind::Mic),
            system: StreamStatus::stopped(StreamKind::System),
            missing_models,
            recording: false,
            rtf: None,
            echo_suppressed: 0,
        }
    }
}

/// Both capture sources, started and stopped together.
pub struct Capture {
    mic: MicSource,
    system: SystemSource,
}

/// Streams as they came up. Either may fail on its own: a working microphone
/// with a broken tap is still worth reporting rather than aborting.
pub struct StartedStreams {
    pub mic: Result<StreamHandle, StreamStatus>,
    pub system: Result<StreamHandle, StreamStatus>,
}

impl StartedStreams {
    pub fn any_running(&self) -> bool {
        self.mic.is_ok() || self.system.is_ok()
    }
}

impl Capture {
    pub fn new() -> Self {
        Self {
            mic: MicSource::default_device(),
            system: SystemSource::new(),
        }
    }

    pub fn start(&mut self) -> StartedStreams {
        StartedStreams {
            mic: self
                .mic
                .start()
                .map_err(|e| StreamStatus::failed(StreamKind::Mic, e.to_string())),
            system: self
                .system
                .start()
                .map_err(|e| StreamStatus::failed(StreamKind::System, e.to_string())),
        }
    }

    pub fn stop(&mut self) {
        self.mic.stop();
        self.system.stop();
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
