//! Decode a list of WAV files with one model and print `file<TAB>text`.
//!
//!     cargo run --release -p shi-pipeline --example transcribe_files -- \
//!         models/<dir> fixtures/mixed-ru-en/*.wav
//!
//! One file is one utterance, so there is no voice-activity detector and no
//! segmentation to disagree about: every model sees exactly the same audio and
//! returns exactly one hypothesis per file. That is what makes the outputs
//! combinable afterwards.

use std::path::PathBuf;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use shi_pipeline::{ModelPaths, Transcriber, load_recognizer, preprocess};

/// A transducer decoded the way `DECODING` says, rather than the way the
/// pipeline does.
///
/// The pipeline always decodes greedily. Whether a beam buys anything on this
/// vocabulary was never measured — the hotword experiment happened to print a
/// beam-search row and it read differently from the greedy one, which is a
/// question, not an answer. `BEAM` sets the number of active paths.
fn decoder_with(dir: &std::path::Path, method: &str, paths: i32) -> OfflineRecognizer {
    let part = |stem: &str| {
        [".int8.onnx", ".onnx"]
            .iter()
            .map(|suffix| dir.join(format!("{stem}{suffix}")))
            .find(|p| p.is_file())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| panic!("no {stem} in {}", dir.display()))
    };
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(part("encoder")),
        decoder: Some(part("decoder")),
        joiner: Some(part("joiner")),
    };
    config.model_config.tokens = Some(dir.join("tokens.txt").to_string_lossy().into());
    config.model_config.model_type = Some("nemo_transducer".into());
    config.model_config.num_threads = 4;
    config.decoding_method = Some(method.to_string());
    config.max_active_paths = paths;
    OfflineRecognizer::create(&config).expect("build the recogniser with that decoding method")
}

fn detect(dir: &std::path::Path) -> Option<ModelPaths> {
    let has = |name: &str| dir.join(name).is_file();

    // Layout before name: T-one ships one model.onnx and a thirty-five entry
    // token list, and its own directory is called "…-streaming-t-one-…", which
    // the name check below would otherwise claim.
    if has("model.onnx") && has("tokens.txt") && !has("encoder.onnx") && !has("encoder.int8.onnx") {
        return Some(ModelPaths::tone_ctc(dir));
    }

    if has("tokens.txt")
        && dir
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("streaming"))
    {
        // Nemotron ships the same four files as Parakeet but needs the online
        // decoder. Nothing inside the directory says so; in the app the model
        // catalogue declares it, and here the name is all there is.
        return Some(ModelPaths::streaming_transducer(dir, Some("auto")));
    }
    if has("conv_frontend.onnx") {
        return Some(ModelPaths::qwen3(dir));
    }
    if has("model.int8.onnx") && has("tokens.txt") {
        return Some(ModelPaths::omnilingual_ctc(dir));
    }
    if has("tokens.txt") {
        return Some(ModelPaths::nemo_transducer(dir));
    }
    let stem = std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        e.file_name().to_string_lossy().strip_suffix("-tokens.txt").map(str::to_string)
    })?;
    Some(ModelPaths::whisper_int8(dir, &stem))
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: transcribe_files <model dir> <wav>..."));
    let files: Vec<PathBuf> = args.map(PathBuf::from).collect();

    let paths = detect(&dir).expect("unrecognised model layout");

    let decoding = std::env::var("DECODING").ok();
    let beam: i32 = std::env::var("BEAM").ok().and_then(|b| b.parse().ok()).unwrap_or(4);
    let custom = decoding.as_deref().map(|method| {
        eprintln!("decoding with {method}, {beam} active paths");
        decoder_with(&dir, method, beam)
    });
    let model = if custom.is_some() {
        None
    } else {
        Some(load_recognizer(&paths, 4).expect("load model"))
    };

    let denoiser = std::env::var("DENOISER").ok().map(|path| {
        preprocess::Denoiser::load(&path, 4).expect("load denoiser")
    });

    for file in files {
        let mut reader = hound::WavReader::open(&file).expect("open wav");
        let rate = reader.spec().sample_rate;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect();
        // PRE names the stages to run, so a preprocessing step can be
        // measured on its own rather than as part of a bundle.
        let stages = std::env::var("PRE").unwrap_or_default();
        let mut samples = samples;
        if stages.contains("dc") {
            preprocess::remove_dc(&mut samples);
        }
        if stages.contains("norm") {
            preprocess::normalize(&mut samples);
        }
        if stages.contains("denoise")
            && let Some(denoiser) = denoiser.as_ref()
        {
            denoiser.run(&mut samples);
        }

        let text = match (&custom, &model) {
            (Some(recogniser), _) => {
                let stream = recogniser.create_stream();
                stream.accept_waveform(rate as i32, &samples);
                recogniser.decode(&stream);
                stream.get_result().map(|r| r.text).unwrap_or_default()
            }
            (None, Some(model)) => model
                .transcribe(&samples)
                .map(|t| t.text)
                .unwrap_or_else(|err| format!("<failed: {err}>")),
            (None, None) => unreachable!("one of the two decoders is always built"),
        };
        println!(
            "{}\t{}",
            file.file_name().unwrap_or_default().to_string_lossy(),
            text.replace('\t', " ").replace('\n', " ")
        );
    }
}
