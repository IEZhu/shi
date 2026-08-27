use std::path::{Path, PathBuf};
use std::time::Duration;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
    OfflineWhisperModelConfig,
};

use crate::error::{PipelineError, Result};

/// Audio must reach the recogniser at this rate; see `shi_audio::TARGET_SAMPLE_RATE`.
pub const ASR_SAMPLE_RATE: i32 = 16_000;

/// Where a recogniser's files are, and which sherpa configuration they need.
///
/// An enum rather than a bag of optional paths: the two families need
/// genuinely different configuration, and a struct with four `Option`s would
/// let a caller build a combination that cannot work.
#[derive(Debug, Clone)]
pub enum ModelPaths {
    /// NeMo transducer — encoder, decoder and joiner, as Parakeet ships.
    NemoTransducer {
        encoder: PathBuf,
        decoder: PathBuf,
        joiner: PathBuf,
        tokens: PathBuf,
    },
    /// Whisper — encoder and decoder only.
    Whisper {
        encoder: PathBuf,
        decoder: PathBuf,
        tokens: PathBuf,
        /// Fixed language, or `None` to let Whisper detect one per utterance.
        language: Option<String>,
    },
}

impl ModelPaths {
    /// A directory laid out the way sherpa-onnx ships Parakeet.
    pub fn parakeet_int8(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self::NemoTransducer {
            encoder: dir.join("encoder.int8.onnx"),
            decoder: dir.join("decoder.int8.onnx"),
            joiner: dir.join("joiner.int8.onnx"),
            tokens: dir.join("tokens.txt"),
        }
    }

    /// A directory laid out the way sherpa-onnx ships Whisper, whose files
    /// carry the size as a prefix: `turbo-encoder.int8.onnx` and so on.
    pub fn whisper_int8(dir: impl AsRef<Path>, prefix: &str) -> Self {
        let dir = dir.as_ref();
        Self::Whisper {
            encoder: dir.join(format!("{prefix}-encoder.int8.onnx")),
            decoder: dir.join(format!("{prefix}-decoder.int8.onnx")),
            tokens: dir.join(format!("{prefix}-tokens.txt")),
            language: None,
        }
    }

    fn files(&self) -> Vec<(&'static str, &PathBuf)> {
        match self {
            ModelPaths::NemoTransducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => vec![
                ("encoder", encoder),
                ("decoder", decoder),
                ("joiner", joiner),
                ("tokens", tokens),
            ],
            ModelPaths::Whisper {
                encoder,
                decoder,
                tokens,
                ..
            } => vec![("encoder", encoder), ("decoder", decoder), ("tokens", tokens)],
        }
    }

    /// Fail before handing paths to the C library, which reports a missing
    /// file as a null recogniser without saying which one was missing.
    pub fn verify(&self) -> Result<()> {
        for (label, path) in self.files() {
            if !path.is_file() {
                return Err(PipelineError::ModelFileMissing {
                    label,
                    path: path.clone(),
                });
            }
        }
        Ok(())
    }

    /// The directory the model lives in, used as its recorded identity.
    fn directory(&self) -> Option<&Path> {
        self.files().first().and_then(|(_, path)| path.parent())
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
        config.model_config.num_threads = threads.max(1);

        match paths {
            ModelPaths::NemoTransducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => {
                config.model_config.transducer = OfflineTransducerModelConfig {
                    encoder: Some(encoder.to_string_lossy().into_owned()),
                    decoder: Some(decoder.to_string_lossy().into_owned()),
                    joiner: Some(joiner.to_string_lossy().into_owned()),
                };
                config.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
                config.model_config.model_type = Some("nemo_transducer".into());
            }
            ModelPaths::Whisper {
                encoder,
                decoder,
                tokens,
                language,
            } => {
                config.model_config.whisper = OfflineWhisperModelConfig {
                    encoder: Some(encoder.to_string_lossy().into_owned()),
                    decoder: Some(decoder.to_string_lossy().into_owned()),
                    language: language.clone(),
                    task: Some("transcribe".into()),
                    // Timestamps are what let the transcript line up with the
                    // audio, so ask for them rather than accepting text alone.
                    enable_token_timestamps: true,
                    ..OfflineWhisperModelConfig::default()
                };
                config.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
                config.model_config.model_type = Some("whisper".into());
            }
        }

        let recognizer =
            OfflineRecognizer::create(&config).ok_or(PipelineError::RecognizerCreateFailed)?;

        let model_id = paths
            .directory()
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "recognizer".into());

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
