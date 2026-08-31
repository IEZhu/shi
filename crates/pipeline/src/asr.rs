use std::path::{Path, PathBuf};
use std::time::Duration;

use sherpa_onnx::{
    OfflineOmnilingualAsrCtcModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer,
    OfflineRecognizerConfig, OfflineTransducerModelConfig, OfflineWhisperModelConfig,
};

use crate::error::{PipelineError, Result};

/// Audio must reach the recogniser at this rate; see `shi_audio::TARGET_SAMPLE_RATE`.
pub const ASR_SAMPLE_RATE: i32 = 16_000;

/// Where a recogniser's files are, and which sherpa configuration they need.
///
/// An enum rather than a bag of optional paths: the two families need
/// genuinely different configuration, and a struct with four `Option`s would
/// What a model part looks like on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    File,
    Directory,
}

impl Expect {
    fn satisfied_by(self, path: &Path) -> bool {
        match self {
            Expect::File => path.is_file(),
            Expect::Directory => path.is_dir(),
        }
    }
}

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
    /// Qwen3-ASR — a speech-conditioned language model. Carries a tokenizer
    /// directory rather than a token list, and a convolutional frontend ahead
    /// of the encoder.
    Qwen3 {
        conv_frontend: PathBuf,
        encoder: PathBuf,
        decoder: PathBuf,
        tokenizer: PathBuf,
    },
    /// Omnilingual ASR — one CTC model, one token list, no separate decoder.
    OmnilingualCtc { model: PathBuf, tokens: PathBuf },
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

    /// A NeMo transducer directory, whichever precision each part happens to
    /// be in.
    ///
    /// Packages are not consistent: Parakeet quantises all three parts, the
    /// multilingual fast-conformer quantises none, and GigaAM ships a
    /// quantised encoder beside a full-precision decoder and joiner. Insisting
    /// on one suffix throughout rejects models that work perfectly well, so
    /// each file is resolved on its own, preferring the smaller one.
    pub fn nemo_transducer(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let part = |stem: &str| {
            [".int8.onnx", ".onnx"]
                .iter()
                .map(|suffix| dir.join(format!("{stem}{suffix}")))
                .find(|path| path.is_file())
                // Nothing on disk: hand back the conventional name so
                // `verify` can name the file that is missing.
                .unwrap_or_else(|| dir.join(format!("{stem}.onnx")))
        };
        Self::NemoTransducer {
            encoder: part("encoder"),
            decoder: part("decoder"),
            joiner: part("joiner"),
            tokens: dir.join("tokens.txt"),
        }
    }

    /// A directory laid out the way sherpa-onnx ships Qwen3-ASR.
    pub fn qwen3(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self::Qwen3 {
            conv_frontend: dir.join("conv_frontend.onnx"),
            encoder: dir.join("encoder.int8.onnx"),
            decoder: dir.join("decoder.int8.onnx"),
            tokenizer: dir.join("tokenizer"),
        }
    }

    /// A directory laid out the way sherpa-onnx ships Omnilingual ASR.
    pub fn omnilingual_ctc(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self::OmnilingualCtc {
            model: dir.join("model.int8.onnx"),
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

    /// What a part is on disk. Qwen3 carries a tokenizer *directory*, and
    /// checking it with `is_file` reported a perfectly good model as missing.
    fn files(&self) -> Vec<(&'static str, &PathBuf, Expect)> {
        match self {
            ModelPaths::NemoTransducer {
                encoder,
                decoder,
                joiner,
                tokens,
            } => vec![
                ("encoder", encoder, Expect::File),
                ("decoder", decoder, Expect::File),
                ("joiner", joiner, Expect::File),
                ("tokens", tokens, Expect::File),
            ],
            ModelPaths::Qwen3 {
                conv_frontend,
                encoder,
                decoder,
                tokenizer,
            } => vec![
                ("conv_frontend", conv_frontend, Expect::File),
                ("encoder", encoder, Expect::File),
                ("decoder", decoder, Expect::File),
                ("tokenizer", tokenizer, Expect::Directory),
            ],
            ModelPaths::OmnilingualCtc { model, tokens } => {
                vec![("model", model, Expect::File), ("tokens", tokens, Expect::File)]
            }
            ModelPaths::Whisper {
                encoder,
                decoder,
                tokens,
                ..
            } => vec![
                ("encoder", encoder, Expect::File),
                ("decoder", decoder, Expect::File),
                ("tokens", tokens, Expect::File),
            ],
        }
    }

    /// Fail before handing paths to the C library, which reports a missing
    /// file as a null recogniser without saying which one was missing.
    pub fn verify(&self) -> Result<()> {
        for (label, path, expect) in self.files() {
            if !expect.satisfied_by(path) {
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
        self.files().first().and_then(|(_, path, _)| path.parent())
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
            ModelPaths::Qwen3 {
                conv_frontend,
                encoder,
                decoder,
                tokenizer,
            } => {
                config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
                    conv_frontend: Some(conv_frontend.to_string_lossy().into_owned()),
                    encoder: Some(encoder.to_string_lossy().into_owned()),
                    decoder: Some(decoder.to_string_lossy().into_owned()),
                    tokenizer: Some(tokenizer.to_string_lossy().into_owned()),
                    // Greedy: a meeting transcript wants the likeliest reading,
                    // not a sampled one, and it has to be reproducible.
                    temperature: 0.0,
                    // The defaults of 128 new tokens and 512 total truncate a
                    // long utterance mid-sentence — the library says so on
                    // stderr and then returns the stump. A minute of speech is
                    // well under a thousand tokens.
                    max_new_tokens: 1024,
                    max_total_len: 2048,
                    ..OfflineQwen3ASRModelConfig::default()
                };
            }
            ModelPaths::OmnilingualCtc { model, tokens } => {
                config.model_config.omnilingual = OfflineOmnilingualAsrCtcModelConfig {
                    model: Some(model.to_string_lossy().into_owned()),
                };
                config.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory holding exactly the files named, each one byte long.
    fn laid_out(name: &str, files: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("shi-asr-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create model dir");
        for file in files {
            std::fs::write(dir.join(file), b"x").expect("write model file");
        }
        dir
    }

    fn parts(paths: &ModelPaths) -> Vec<String> {
        paths
            .files()
            .into_iter()
            .map(|(_, path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn a_fully_quantised_package_is_recognised() {
        let dir = laid_out(
            "quantised",
            &["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
        );
        let paths = ModelPaths::nemo_transducer(&dir);
        assert!(paths.verify().is_ok());
        assert_eq!(
            parts(&paths),
            ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
        );
    }

    #[test]
    fn a_full_precision_package_is_recognised() {
        let dir = laid_out(
            "full",
            &["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        );
        let paths = ModelPaths::nemo_transducer(&dir);
        assert!(paths.verify().is_ok());
        assert_eq!(
            parts(&paths),
            ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        );
    }

    #[test]
    fn a_package_mixing_precisions_is_recognised() {
        // GigaAM ships exactly this: a quantised encoder, everything else full.
        // Requiring one suffix throughout would reject a working model.
        let dir = laid_out(
            "mixed",
            &["encoder.int8.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
        );
        let paths = ModelPaths::nemo_transducer(&dir);
        assert!(paths.verify().is_ok());
        assert_eq!(
            parts(&paths),
            ["encoder.int8.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        );
    }

    #[test]
    fn a_tokenizer_directory_counts_as_present() {
        // Qwen3 ships a tokenizer *directory*. Checking every part with
        // `is_file` reported a complete model as missing one, and the load
        // failed before sherpa-onnx ever saw it.
        let dir = laid_out("qwen3", &["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"]);
        std::fs::create_dir_all(dir.join("tokenizer")).expect("tokenizer dir");
        assert!(ModelPaths::qwen3(&dir).verify().is_ok());
    }

    #[test]
    fn a_tokenizer_that_is_a_plain_file_is_rejected() {
        let dir = laid_out(
            "qwen3-flat",
            &["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer"],
        );
        let error = ModelPaths::qwen3(&dir).verify().unwrap_err();
        assert!(format!("{error}").contains("tokenizer"));
    }

    #[test]
    fn a_single_file_ctc_model_is_recognised() {
        let dir = laid_out("omnilingual", &["model.int8.onnx", "tokens.txt"]);
        assert!(ModelPaths::omnilingual_ctc(&dir).verify().is_ok());
    }

    #[test]
    fn a_missing_file_is_named_rather_than_guessed_at() {
        let dir = laid_out("incomplete", &["encoder.onnx", "decoder.onnx", "tokens.txt"]);
        let error = ModelPaths::nemo_transducer(&dir).verify().unwrap_err();
        assert!(
            format!("{error}").contains("joiner"),
            "the error must say which file is missing, said: {error}"
        );
    }
}
