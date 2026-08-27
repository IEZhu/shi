use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("model file missing: {label} at {}", path.display())]
    ModelFileMissing {
        label: &'static str,
        path: PathBuf,
    },

    #[error("sherpa-onnx refused to build a recognizer — check the model files match the declared model type")]
    RecognizerCreateFailed,

    #[error("sherpa-onnx refused to build a voice activity detector")]
    VadCreateFailed,

    #[error("sherpa-onnx refused to load the speaker embedding model")]
    SpeakerModelLoadFailed,

    #[error("cannot resample {from} Hz to {to} Hz")]
    ResamplerCreateFailed { from: u32, to: u32 },
}

pub type Result<T> = std::result::Result<T, PipelineError>;
