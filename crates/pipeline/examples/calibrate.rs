//! Pick a speaker-embedding model and measure the thresholds it needs.
//!
//! The plan refused to hard-code a cosine threshold for voice identification,
//! because the right value depends on the model, the microphones and the room.
//! This is the harness that produces one: give it a directory of clips named
//! `<speaker>_<n>.wav` and it reports, per model, how far apart the same voice
//! and different voices actually land.
//!
//!     cargo run -p shi-pipeline --release --example calibrate -- <corpus-dir> <model.onnx>...

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, Wave};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

struct Stats {
    min: f32,
    mean: f32,
    max: f32,
}

fn stats(values: &[f32]) -> Stats {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    Stats {
        min: values.iter().copied().fold(f32::MAX, f32::min),
        mean,
        max: values.iter().copied().fold(f32::MIN, f32::max),
    }
}

/// Equal-error threshold: where a same-voice pair is as likely to be rejected
/// as a different-voice pair is to be accepted.
fn equal_error(same: &[f32], different: &[f32]) -> (f32, f32) {
    let mut best = (0.0f32, 1.0f32);
    let mut threshold = 0.0f32;
    while threshold <= 1.0 {
        let false_reject =
            same.iter().filter(|s| **s < threshold).count() as f32 / same.len() as f32;
        let false_accept =
            different.iter().filter(|d| **d >= threshold).count() as f32 / different.len() as f32;
        let gap = (false_reject - false_accept).abs();
        if gap < best.1 {
            best = (threshold, gap);
        }
        threshold += 0.005;
    }
    let rate = same.iter().filter(|s| **s < best.0).count() as f32 / same.len() as f32;
    (best.0, rate)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(args.next().expect("usage: calibrate <corpus-dir> <model>..."));
    let models: Vec<PathBuf> = args.map(PathBuf::from).collect();

    // Clips are named <speaker>_<n>.wav.
    let mut clips: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in std::fs::read_dir(&corpus).expect("read corpus").flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "wav") {
            continue;
        }
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let speaker = stem.rsplit_once('_').map(|(s, _)| s).unwrap_or(&stem);
        clips.entry(speaker.to_string()).or_default().push(path);
    }

    let total: usize = clips.values().map(|v| v.len()).sum();
    println!("corpus: {total} clips from {} voices\n", clips.len());

    for model in &models {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            num_threads: 2,
            debug: false,
            provider: None,
        };
        let Some(extractor) = SpeakerEmbeddingExtractor::create(&config) else {
            eprintln!("cannot load {}", model.display());
            continue;
        };

        let started = Instant::now();
        let mut embeddings: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
        for (speaker, paths) in &clips {
            for path in paths {
                let Some(wave) = Wave::read(path.to_string_lossy().as_ref()) else {
                    continue;
                };
                let Some(stream) = extractor.create_stream() else {
                    continue;
                };
                stream.accept_waveform(wave.sample_rate(), wave.samples());
                stream.input_finished();
                if let Some(embedding) = extractor.compute(&stream) {
                    embeddings.entry(speaker.clone()).or_default().push(embedding);
                }
            }
        }
        let elapsed = started.elapsed();

        let mut same = Vec::new();
        let mut different = Vec::new();
        // Aggregate numbers hide the case that matters: the two voices this
        // model finds hardest to tell apart. That pair, not the average, is
        // what a threshold has to survive.
        let mut worst_pairs: Vec<(f32, String)> = Vec::new();
        let mut weakest_self: Vec<(f32, String)> = Vec::new();
        let names: Vec<&String> = embeddings.keys().collect();

        for (i, a) in names.iter().enumerate() {
            let list_a = &embeddings[*a];
            let mut self_scores = Vec::new();
            for x in 0..list_a.len() {
                for y in (x + 1)..list_a.len() {
                    let score = cosine(&list_a[x], &list_a[y]);
                    same.push(score);
                    self_scores.push(score);
                }
            }
            if let Some(min) = self_scores.iter().copied().reduce(f32::min) {
                weakest_self.push((min, (*a).clone()));
            }

            for b in names.iter().skip(i + 1) {
                let mut pair_max = f32::MIN;
                for ea in list_a {
                    for eb in &embeddings[*b] {
                        let score = cosine(ea, eb);
                        different.push(score);
                        pair_max = pair_max.max(score);
                    }
                }
                worst_pairs.push((pair_max, format!("{a} vs {b}")));
            }
        }

        worst_pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        weakest_self.sort_by(|x, y| x.0.total_cmp(&y.0));

        if same.is_empty() || different.is_empty() {
            eprintln!("{}: not enough pairs", model.display());
            continue;
        }

        let s = stats(&same);
        let d = stats(&different);
        let (threshold, eer) = equal_error(&same, &different);
        let dim = extractor.dim();

        println!("{}", model.file_name().unwrap_or_default().to_string_lossy());
        println!("  dim {dim}, {total} clips in {:.1}s", elapsed.as_secs_f32());
        println!(
            "  same voice       min {:.3}  mean {:.3}  max {:.3}   ({} pairs)",
            s.min, s.mean, s.max, same.len()
        );
        println!(
            "  different voice  min {:.3}  mean {:.3}  max {:.3}   ({} pairs)",
            d.min, d.mean, d.max, different.len()
        );
        println!(
            "  margin {:+.3}   equal-error threshold {:.3} at {:.1}%",
            s.min - d.max,
            threshold,
            eer * 100.0
        );
        println!("  hardest to separate:");
        for (score, pair) in worst_pairs.iter().take(3) {
            println!("    {score:.3}  {pair}");
        }
        println!("  weakest self-match:");
        for (score, name) in weakest_self.iter().take(2) {
            println!("    {score:.3}  {name}");
        }
        println!();
    }
}
