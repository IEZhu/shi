/// Anything that can stop the app doing what the user asked.
///
/// Serialises to a plain string for the frontend: Tauri commands surface
/// errors as rejected promises, and the UI shows the message verbatim.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("models not downloaded: {}", .0.join(", "))]
    ModelsMissing(Vec<String>),

    #[error(transparent)]
    Store(#[from] shi_store::StoreError),

    #[error(transparent)]
    Pipeline(#[from] shi_pipeline::PipelineError),

    #[error("a speaker needs a name")]
    EmptyName,

    #[error("unknown model: {0}")]
    UnknownModel(String),

    #[error(transparent)]
    Models(#[from] shi_models::ModelError),

    #[error("cannot change settings while a meeting is running")]
    BusyRecording,

    #[error("cannot create application directories: {0}")]
    Io(#[from] std::io::Error),
}

impl serde::Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}
