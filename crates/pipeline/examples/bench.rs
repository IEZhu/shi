//! Measure what transcription actually costs on this machine.
//!
//! The plan treats real-time factor as a gate rather than an assumption: the
//! published RTFx figures for Parakeet come from batched GPU runs, and the
//! draft-text cadence is only affordable if a single CPU core is fast enough.
//! This prints the number the `Cadence` controller will be working with, and
//! where it settles.
//!
//!     cargo run -p shi-pipeline --release --example bench -- <model-dir> <wav>...

use std::path::PathBuf;
use std::time::{Duration, Instant};

use sherpa_onnx::Wave;
use shi_pipeline::{Cadence, ModelPaths, SherpaTranscriber, Transcriber};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8".into()),
    );
    let wavs: Vec<PathBuf> = args.map(PathBuf::from).collect();
    let wavs = if wavs.is_empty() {
        std::fs::read_dir(model_dir.join("test_wavs"))
            .map(|entries| {
                let mut found: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                    .collect();
                found.sort();
                found
            })
            .unwrap_or_default()
    } else {
        wavs
    };

    if wavs.is_empty() {
        eprintln!("no wav files to benchmark");
        std::process::exit(1);
    }

    let paths = ModelPaths::parakeet_int8(&model_dir);
    let cores = std::thread::available_parallelism().map_or(8, |n| n.get());

    println!("model:  {}", model_dir.display());
    println!("cores:  {cores}");
    println!("clips:  {}\n", wavs.len());

    for threads in [1, 2, 4, 8] {
        if threads > cores as i32 {
            continue;
        }

        let loading = Instant::now();
        let transcriber = match SherpaTranscriber::load(&paths, threads) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("load failed: {e}");
                std::process::exit(1);
            }
        };
        let load_time = loading.elapsed();

        let mut audio_total = Duration::ZERO;
        let mut decode_total = Duration::ZERO;
        let mut first_text = String::new();

        for wav in &wavs {
            let Some(wave) = Wave::read(wav.to_string_lossy().as_ref()) else {
                eprintln!("cannot read {}", wav.display());
                continue;
            };
            let samples = wave.samples();
            let audio = Duration::from_secs_f64(samples.len() as f64 / wave.sample_rate() as f64);

            // Discard the first pass: it pays for lazy allocation inside ORT.
            let _ = transcriber.transcribe(samples);

            let started = Instant::now();
            let transcript = transcriber.transcribe(samples).expect("transcribe");
            let decode = started.elapsed();

            audio_total += audio;
            decode_total += decode;
            if first_text.is_empty() {
                first_text = transcript.text.clone();
            }
        }

        let rtf = decode_total.as_secs_f32() / audio_total.as_secs_f32();
        let mut cadence = Cadence::default();
        for _ in 0..30 {
            let window = cadence.max_window();
            cadence.observe(window, Duration::from_secs_f32(window.as_secs_f32() * rtf));
        }

        println!(
            "threads={threads}  load={:.1}s  rtf={rtf:.4}  RTFx={:.1}  ->  drafts every {:?}, window {:?}",
            load_time.as_secs_f32(),
            1.0 / rtf,
            cadence.interval(),
            cadence.max_window(),
        );
    }

    println!("\nsample output: {first_sample}", first_sample = "see below");
    let transcriber = SherpaTranscriber::load(&paths, 4).expect("load");
    for wav in wavs.iter().take(4) {
        if let Some(wave) = Wave::read(wav.to_string_lossy().as_ref()) {
            let t = transcriber.transcribe(wave.samples()).expect("transcribe");
            println!(
                "  {:<10} {} tokens, {} timestamps\n             {}",
                wav.file_name().unwrap_or_default().to_string_lossy(),
                t.tokens.len(),
                t.token_offsets.len(),
                t.text
            );
        }
    }
}
