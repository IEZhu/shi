use crate::error::{AudioError, Result};
use crate::source::{AudioSource, StreamHandle, StreamKind};

/// Placeholder until the Windows (WASAPI loopback) and Linux (PipeWire
/// monitor) backends land. It fails loudly rather than silently recording
/// nothing, so a half-finished port cannot masquerade as a working one.
pub struct UnsupportedSystemSource;

impl UnsupportedSystemSource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UnsupportedSystemSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for UnsupportedSystemSource {
    fn start(&mut self) -> Result<StreamHandle> {
        Err(AudioError::Unsupported(format!(
            "system audio capture is not implemented for {} yet",
            std::env::consts::OS
        )))
    }

    fn stop(&mut self) {}

    fn kind(&self) -> StreamKind {
        StreamKind::System
    }
}
