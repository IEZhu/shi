//! The streaming recogniser, decoded one utterance at a time.
//!
//! Needs the model, which is half a gigabyte and not in the repository; the
//! test says so and passes when it is absent, the way the other model-backed
//! tests here do.

use std::path::{Path, PathBuf};

use shi_pipeline::{ModelPaths, StreamingTranscriber, Transcriber};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

fn model() -> Option<PathBuf> {
    let dir = repo_root()
        .join("models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11");
    dir.join("encoder.int8.onnx").is_file().then_some(dir)
}

fn samples(name: &str) -> Vec<f32> {
    let mut reader =
        hound::WavReader::open(repo_root().join("fixtures/mixed-ru-en").join(name)).expect("open");
    reader.samples::<i16>().map(|s| s.expect("sample") as f32 / i16::MAX as f32).collect()
}

#[test]
fn an_utterance_is_not_clipped_at_either_end() {
    let Some(dir) = model() else {
        eprintln!("skipping: the streaming model is not installed");
        return;
    };

    // A cache-aware model spends its first chunk warming up and needs another
    // to flush the last of the audio. Without padding this exact sentence came
    // back as "Broker lost its leader partition and the consumer group" —
    // missing its first word and its last.
    let paths = ModelPaths::streaming_transducer(&dir, Some("auto"));
    let model = StreamingTranscriber::load(&paths, 4).expect("load streaming model");
    let text = model.transcribe(&samples("03-en.wav")).expect("decode").text;
    let words = text.to_lowercase();

    assert!(
        words.starts_with("the broker"),
        "the opening words were clipped: {text:?}"
    );
    assert!(
        words.trim_end_matches('.').ends_with("rebalanced"),
        "the closing word was clipped: {text:?}"
    );
}
