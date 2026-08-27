/// Failures a capture source can report.
///
/// `PermissionDenied` is deliberately its own variant rather than a string:
/// the readiness panel needs to distinguish "you must grant access" (an
/// actionable prompt) from "the device broke" (a retry).
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no {0} device available")]
    NoDevice(&'static str),

    #[error("{0} capture permission denied — grant it in System Settings > Privacy & Security")]
    PermissionDenied(&'static str),

    #[error("device error: {0}")]
    Device(String),

    #[error("{context} failed (OSStatus {status})")]
    Platform { context: &'static str, status: i32 },

    #[error("unsupported audio configuration: {0}")]
    Unsupported(String),

    #[error("source already running")]
    AlreadyRunning,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AudioError>;
