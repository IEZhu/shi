//! Ask the recording whether its voices are separable at all.
//!
//!     cargo run --release -p shi-pipeline --example diarize_sweep -- <base>
//!
//! Embeds every utterance once, then sweeps the clustering threshold. If no
//! threshold yields a sane number of speakers, the problem is the embeddings —
//! not the threshold — and no amount of tuning will fix it.

use std::time::Instant;

use shi_pipeline::{Span, SpeakerTracker, Thresholds, cluster};

fn main() {
    let base = std::env::args().nth(1).expect("usage: diarize_sweep <base>");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    let raw = std::fs::read(format!("{base}.system16k.f32")).expect("read kept audio");
    let audio: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!("audio: {:.1} min at 16 kHz", audio.len() as f32 / 16_000.0 / 60.0);

    let tsv = std::fs::read_to_string(format!("{base}.spans.tsv")).expect("read spans");
    let spans: Vec<Span> = tsv
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut f = line.split('\t');
            let id: i64 = f.next()?.parse().ok()?;
            let start_ms: i64 = f.next()?.parse().ok()?;
            let end_ms: i64 = f.next()?.parse().ok()?;
            (id >= 0 && end_ms > start_ms).then_some(Span { id, start_ms, end_ms })
        })
        .collect();
    println!("system utterances: {}", spans.len());

    let tracker = SpeakerTracker::new(
        &format!("{root}/models/nemo_en_titanet_small.onnx"),
        4,
        Thresholds::default(),
    )
    .expect("load speaker model");

    let began = Instant::now();
    let mut embeddings = Vec::new();
    let mut durations = Vec::new();
    let mut skipped = 0;
    for span in &spans {
        let from = (span.start_ms as usize * 16) .min(audio.len());
        let to = (span.end_ms as usize * 16).min(audio.len());
        if to <= from || (to - from) < 16_000 * 7 / 10 {
            skipped += 1;
            continue;
        }
        match tracker.embed(&audio[from..to]) {
            Some(e) => {
                embeddings.push(e);
                durations.push((to - from) as f32 / 16_000.0);
            }
            None => skipped += 1,
        }
    }
    println!(
        "embedded {} utterances in {:.1}s ({} too short or refused)\n",
        embeddings.len(),
        began.elapsed().as_secs_f32(),
        skipped
    );

    // How similar are utterances to each other at all? If everything looks
    // alike, or nothing does, clustering cannot help.
    let mut sims = Vec::new();
    for a in 0..embeddings.len() {
        for b in a + 1..embeddings.len() {
            sims.push(cosine(&embeddings[a], &embeddings[b]));
        }
    }
    sims.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let pct = |p: f32| sims[((sims.len() as f32 - 1.0) * p) as usize];
    println!(
        "pairwise cosine: min {:.3}  p10 {:.3}  median {:.3}  p90 {:.3}  max {:.3}",
        sims[0], pct(0.1), pct(0.5), pct(0.9), sims[sims.len() - 1]
    );

    println!("\nthreshold  clusters  largest clusters by speech time");
    for step in 0..=10 {
        let threshold = 0.40 + step as f32 * 0.05;
        let labels = cluster(&embeddings, threshold);
        let count = labels.iter().collect::<std::collections::BTreeSet<_>>().len();
        let mut time = std::collections::BTreeMap::new();
        for (label, seconds) in labels.iter().zip(&durations) {
            *time.entry(*label).or_insert(0.0f32) += seconds;
        }
        let mut top: Vec<f32> = time.values().copied().collect();
        top.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let shown: Vec<String> = top.iter().take(6).map(|s| format!("{s:.0}s")).collect();
        println!("   {threshold:.2}     {count:>4}      {}", shown.join(" "));
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na * nb <= f32::EPSILON { 0.0 } else { dot / (na * nb) }
}
