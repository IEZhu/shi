use std::path::{Path, PathBuf};
use std::time::Duration;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::error::{PipelineError, Result};

/// Audio must reach the recogniser at this rate; see `shi_audio::TARGET_SAMPLE_RATE`.
pub const ASR_SAMPLE_RATE: i32 = 16_000;

/// Files making up a transducer ASR model on disk.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
    /// sherpa's identifier for the architecture, e.g. `nemo_transducer`.
    pub model_type: String,
}

impl ModelPaths {
    /// Locate a model laid out the way sherpa-onnx ships it: one directory
    /// holding `encoder`/`decoder`/`joiner` int8 ONNX files plus `tokens.txt`.
    pub fn parakeet_int8(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            encoder: dir.join("encoder.int8.onnx"),
            decoder: dir.join("decoder.int8.onnx"),
            joiner: dir.join("joiner.int8.onnx"),
            tokens: dir.join("tokens.txt"),
            model_type: "nemo_transducer".into(),
        }
    }

    /// Fail before handing paths to the C library, which reports a missing file
    /// as a null recogniser with no indication of which one was missing.
    pub fn verify(&self) -> Result<()> {
        for (label, path) in [
            ("encoder", &self.encoder),
            ("decoder", &self.decoder),
            ("joiner", &self.joiner),
            ("tokens", &self.tokens),
        ] {
            if !path.is_file() {
                return Err(PipelineError::ModelFileMissing {
                    label,
                    path: path.clone(),
                });
            }
        }
        Ok(())
    }
}

/// One decoded stretch of speech.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<String>,
    /// Offset of each token from the start of the decoded audio, when the
    /// model provides them. Parakeet does; some models do not.
    pub token_offsets: Vec<Duration>,
}

impl Transcript {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Anything that can turn 16 kHz mono audio into text.
///
/// Behind a trait so the engine can be exercised with a stub in tests and so
/// swapping Parakeet for Whisper is a construction detail, not a rewrite.
pub trait Transcriber: Send + Sync {
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript>;
    /// Human-readable model identity, recorded with each meeting.
    fn model_id(&self) -> &str;
}

/// Transducer recogniser backed by sherpa-onnx.
pub struct SherpaTranscriber {
    recognizer: OfflineRecognizer,
    model_id: String,
}

impl SherpaTranscriber {
    pub fn load(paths: &ModelPaths, threads: i32) -> Result<Self> {
        paths.verify()?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(paths.encoder.to_string_lossy().into_owned()),
            decoder: Some(paths.decoder.to_string_lossy().into_owned()),
            joiner: Some(paths.joiner.to_string_lossy().into_owned()),
        };
        config.model_config.tokens = Some(paths.tokens.to_string_lossy().into_owned());
        config.model_config.model_type = Some(paths.model_type.clone());
        config.model_config.num_threads = threads.max(1);

        let recognizer =
            OfflineRecognizer::create(&config).ok_or(PipelineError::RecognizerCreateFailed)?;

        let model_id = paths
            .encoder
            .parent()
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| paths.model_type.clone());

        Ok(Self {
            recognizer,
            model_id,
        })
    }
}

impl Transcriber for SherpaTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        if samples.is_empty() {
            return Ok(Transcript::default());
        }

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(ASR_SAMPLE_RATE, samples);
        self.recognizer.decode(&stream);

        let Some(result) = stream.get_result() else {
            return Ok(Transcript::default());
        };

        Ok(Transcript {
            token_offsets: result
                .timestamps
                .unwrap_or_default()
                .into_iter()
                .map(Duration::from_secs_f32)
                .collect(),
            text: result.text,
            tokens: result.tokens,
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
