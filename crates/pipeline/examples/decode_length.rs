//! How much does the length of one decode cost in accuracy?
//!
//! `docs/transcription.md` measures this by cutting a corpus in the gaps
//! between sentences, which flatters the idea: those gaps exist only because
//! the sentences were recorded separately. This runs the real pipeline instead
//! — the real detector, the real `split_utterance`, cutting at the quietest
//! moment it can find, which is sometimes inside a word.
//!
//!     cargo run --release -p shi-pipeline --example decode_length -- \
//!         target/dev-bundles/concat/clean-whole.wav \
//!         target/dev-bundles/concat/reference.txt \
//!         5000 10000 20000 30000

use std::sync::Arc;
use std::time::{Duration, Instant};

use shi_audio::StreamKind;
use shi_pipeline::{
    ModelPaths, PipelineEvent, SherpaTranscriber, StreamPipeline, Transcriber, VadSettings,
};

/// Words, folded the way `scripts/rover.py` folds them, so the numbers here and
/// the numbers in the document mean the same thing.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace('ё', "е")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn errors(reference: &[String], hypothesis: &[String]) -> usize {
    let (n, m) = (reference.len(), hypothesis.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut row = vec![0usize; m + 1];
    for i in 1..=n {
        row[0] = i;
        for j in 1..=m {
            let same = reference[i - 1] == hypothesis[j - 1];
            row[j] = (prev[j] + 1)
                .min(row[j - 1] + 1)
                .min(prev[j - 1] + usize::from(!same));
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[m]
}

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open the audio");
    let rate = reader.spec().sample_rate;
    let samples = reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect();
    (samples, rate)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = args.next().expect("usage: decode_length <wav> <reference.txt> <ms>...");
    let reference_path = args.next().expect("a reference transcript");
    let caps: Vec<u64> = args.filter_map(|a| a.parse().ok()).collect();
    let caps = if caps.is_empty() { vec![5_000, 10_000, 20_000, 30_000] } else { caps };

    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let silero = format!("{root}/models/silero_vad.onnx");
    let transcriber: Arc<dyn Transcriber> = Arc::new(
        SherpaTranscriber::load(
            &ModelPaths::parakeet_int8(std::path::PathBuf::from(format!(
                "{root}/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
            ))),
            4,
        )
        .expect("load the recogniser"),
    );

    let reference = words(&std::fs::read_to_string(&reference_path).expect("read the reference"));
    let (audio, rate) = read_wav(&wav);
    println!(
        "{:.0} s of audio, {} reference words\n",
        audio.len() as f32 / rate as f32,
        reference.len()
    );
    println!("{:>9} {:>10} {:>7} {:>9} {:>8}", "cap", "utterances", "WER", "words", "seconds");

    for cap in caps {
        let settings = VadSettings {
            longest_decode: Duration::from_millis(cap),
            ..VadSettings::default()
        };
        let mut pipeline =
            StreamPipeline::new(StreamKind::Mic, rate, &silero, Arc::clone(&transcriber), settings)
                .expect("build the pipeline");

        let began = Instant::now();
        let mut said: Vec<String> = Vec::new();
        let mut collect = |events: Vec<PipelineEvent>, into: &mut Vec<String>| {
            for event in events {
                if let PipelineEvent::Final { text, .. } = event
                    && !text.trim().is_empty()
                {
                    into.push(text);
                }
            }
        };
        // A zero clock means the pipeline never decides it is behind and never
        // pads with silence: the file is contiguous and its own sample count is
        // the only timeline there is.
        for block in audio.chunks(rate as usize / 2) {
            collect(pipeline.push_at(block, Duration::ZERO), &mut said);
        }
        collect(pipeline.flush(), &mut said);
        let spent = began.elapsed();

        let heard = words(&said.join(" "));
        let wrong = errors(&reference, &heard);
        println!(
            "{:>7} ms {:>10} {:>6.0}% {:>9} {:>7.1}s",
            cap,
            said.len(),
            100.0 * wrong as f32 / reference.len().max(1) as f32,
            heard.len(),
            spent.as_secs_f32(),
        );
    }
}
