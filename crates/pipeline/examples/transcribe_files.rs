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

use shi_pipeline::{ModelPaths, SherpaTranscriber, Transcriber};

fn detect(dir: &std::path::Path) -> Option<ModelPaths> {
    let has = |name: &str| dir.join(name).is_file();
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
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: transcribe_files <model dir> <wav>..."));
    let files: Vec<PathBuf> = args.map(PathBuf::from).collect();

    let paths = detect(&dir).expect("unrecognised model layout");
    let model = SherpaTranscriber::load(&paths, 4).expect("load model");

    for file in files {
        let mut reader = hound::WavReader::open(&file).expect("open wav");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect();
        let text = model
            .transcribe(&samples)
            .map(|t| t.text)
            .unwrap_or_else(|err| format!("<failed: {err}>"));
        println!(
            "{}\t{}",
            file.file_name().unwrap_or_default().to_string_lossy(),
            text.replace('\t', " ").replace('\n', " ")
        );
    }
}
