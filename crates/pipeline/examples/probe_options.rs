//! Ask a streaming recogniser which per-stream options it actually knows.
//!
//!     cargo run --release -p shi-pipeline --example probe_options -- models/<dir>
//!
//! sherpa-onnx exposes `has_option`, and the option names are not documented
//! anywhere this project could find, so the honest way to learn them is to ask.

use std::path::PathBuf;

use shi_pipeline::{ModelPaths, StreamingTranscriber};

const CANDIDATES: &[&str] = &[
    "language", "lang", "target_lang", "target_language", "src_lang", "source_language",
    "locale", "language_id", "lang_id", "prompt", "prompt_index", "prompt_lang",
    "nemotron_language", "decoder_prompt", "task", "text_prompt", "hotwords",
    "itn", "punct", "timestamps",
];

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: probe_options <model dir>"));
    let paths = ModelPaths::streaming_transducer(&dir, None);
    let model = StreamingTranscriber::load(&paths, 2).expect("load streaming model");

    println!("options this model admits to knowing:");
    let mut any = false;
    for key in CANDIDATES {
        if model.knows_option(key) {
            println!("  {key}  (currently {:?})", model.option(key));
            any = true;
        }
    }
    if !any {
        println!("  none of {} candidates", CANDIDATES.len());
    }
}
