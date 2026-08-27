//! Does contextual biasing actually fix domain vocabulary?
//!
//! The benchmark produced «платежным шлезом» for «платёжным шлюзом» — a
//! product word the model has never seen. sherpa-onnx exposes hotword biasing,
//! but with an undocumented dependency on the decoding method, so this checks
//! rather than assumes.
//!
//!     cargo run -p shi-pipeline --release --example hotwords -- <model-dir> <wav> <word>...

use std::path::PathBuf;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig, Wave,
};

fn transcribe(
    model_dir: &PathBuf,
    wav: &str,
    decoding: Option<&str>,
    hotwords_file: Option<&str>,
    score: f32,
) -> Option<String> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(model_dir.join("encoder.int8.onnx").to_string_lossy().into()),
        decoder: Some(model_dir.join("decoder.int8.onnx").to_string_lossy().into()),
        joiner: Some(model_dir.join("joiner.int8.onnx").to_string_lossy().into()),
    };
    config.model_config.tokens = Some(model_dir.join("tokens.txt").to_string_lossy().into());
    config.model_config.model_type = Some("nemo_transducer".into());
    config.model_config.num_threads = 4;
    config.decoding_method = decoding.map(str::to_string);
    config.hotwords_file = hotwords_file.map(str::to_string);
    config.hotwords_score = score;

    let recognizer = OfflineRecognizer::create(&config)?;
    let wave = Wave::read(wav)?;
    let stream = recognizer.create_stream();
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    recognizer.decode(&stream);
    stream.get_result().map(|r| r.text)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("usage: hotwords <model-dir> <wav> <word>..."));
    let wav = args.next().expect("a wav file");
    let words: Vec<String> = args.collect();

    let hotwords_path = std::env::temp_dir().join("shi-hotwords.txt");
    std::fs::write(&hotwords_path, format!("{}\n", words.join("\n"))).expect("write hotwords");
    let hotwords = hotwords_path.to_string_lossy().into_owned();

    println!("biasing towards: {}\n", words.join(", "));

    let cases: [(&str, Option<&str>, bool, f32); 7] = [
        ("greedy, no hotwords", Some("greedy_search"), false, 0.0),
        ("beam, no hotwords", Some("modified_beam_search"), false, 0.0),
        ("beam + hotwords 1.5", Some("modified_beam_search"), true, 1.5),
        ("beam + hotwords 3", Some("modified_beam_search"), true, 3.0),
        ("beam + hotwords 6", Some("modified_beam_search"), true, 6.0),
        ("beam + hotwords 12", Some("modified_beam_search"), true, 12.0),
        ("beam + hotwords 30", Some("modified_beam_search"), true, 30.0),
    ];

    for (label, decoding, use_hotwords, score) in cases {
        let text = transcribe(
            &model_dir,
            &wav,
            decoding,
            use_hotwords.then_some(hotwords.as_str()),
            score,
        );
        match text {
            Some(text) => println!("{label:<28} {text}"),
            None => println!("{label:<28} <recogniser refused this configuration>"),
        }
    }
}
