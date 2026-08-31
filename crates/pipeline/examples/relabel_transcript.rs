//! Re-label a transcript's speakers from the saved recording, without running
//! recognition again.
//!
//!     cargo run --release -p shi-pipeline --example relabel_transcript -- \
//!         <base> <threshold> <out.md>
//!
//! The text is expensive and does not change; who said it is cheap and, on real
//! audio, wrong at the default threshold. `diarize_sweep` finds the threshold
//! this recording actually wants; this applies it.

use shi_pipeline::{Span, SpeakerTracker, Thresholds, cluster};

struct Line {
    id: i64,
    start_ms: i64,
    end_ms: i64,
    who: String,
    text: String,
}

fn stamp(ms: i64) -> String {
    let total = ms / 1000;
    format!("{:02}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let base = args.next().expect("usage: relabel_transcript <base> <threshold> <out.md>");
    let threshold: f32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0.45);
    let out = args.next().unwrap_or_else(|| "meeting.md".into());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    let raw = std::fs::read(format!("{base}.system16k.f32")).expect("read kept audio");
    let audio: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let tsv = std::fs::read_to_string(format!("{base}.spans.tsv")).expect("read spans");
    let mut lines: Vec<Line> = tsv
        .lines()
        .skip(1)
        .filter_map(|line| {
            let mut f = line.splitn(5, '\t');
            Some(Line {
                id: f.next()?.parse().ok()?,
                start_ms: f.next()?.parse().ok()?,
                end_ms: f.next()?.parse().ok()?,
                who: f.next()?.to_string(),
                text: f.next()?.to_string(),
            })
        })
        .collect();

    let tracker = SpeakerTracker::new(
        &format!("{root}/models/nemo_en_titanet_small.onnx"),
        4,
        Thresholds::default(),
    )
    .expect("load speaker model");

    let mut embedded = Vec::new();
    let mut embeddings = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if line.id < 0 {
            continue; // the microphone: the speaker is known by construction
        }
        let from = (line.start_ms as usize * 16).min(audio.len());
        let to = (line.end_ms as usize * 16).min(audio.len());
        if to <= from {
            continue;
        }
        if let Some(embedding) = tracker.embed(&audio[from..to]) {
            embedded.push(index);
            embeddings.push(embedding);
        }
    }

    // Number speakers by how much they actually said, so "Спикер 1" is the
    // person who did most of the talking rather than whoever happened to
    // speak first.
    let labels = cluster(&embeddings, threshold);
    let mut spoken: std::collections::BTreeMap<usize, i64> = std::collections::BTreeMap::new();
    for (label, index) in labels.iter().zip(&embedded) {
        let line = &lines[*index];
        *spoken.entry(*label).or_insert(0) += line.end_ms - line.start_ms;
    }
    let mut order: Vec<(usize, i64)> = spoken.into_iter().collect();
    order.sort_by_key(|(_, ms)| -*ms);
    let rank: std::collections::BTreeMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(position, (label, _))| (*label, position + 1))
        .collect();

    for (label, index) in labels.iter().zip(&embedded) {
        lines[*index].who = format!("Спикер {}", rank[label]);
    }
    // Utterances the clustering could not judge inherit the voice around them.
    let mut carried: Option<String> = None;
    for line in lines.iter_mut() {
        if line.id < 0 {
            continue;
        }
        match embedded.binary_search(&(line.id as usize)) {
            Ok(_) => carried = Some(line.who.clone()),
            Err(_) => {
                if let Some(who) = &carried {
                    line.who = who.clone();
                }
            }
        }
    }

    let speakers = order.len();
    let mut body = String::new();
    let mut previous = String::new();
    for line in &lines {
        if line.who != previous {
            body.push_str(&format!(
                "\n**[{}] {}:** {}\n",
                stamp(line.start_ms),
                line.who,
                line.text
            ));
            previous = line.who.clone();
        } else {
            body.push_str(&format!("{}\n", line.text));
        }
    }

    let duration = lines.last().map_or(0, |l| l.end_ms.max(l.start_ms));
    let header = format!(
        "---\ntitle: Встреча\nduration: {}\nspeakers: {speakers}\ndiarization_threshold: {threshold}\n---\n",
        stamp(duration)
    );
    std::fs::write(&out, header + &body).expect("write transcript");

    println!("{speakers} speakers at threshold {threshold}, written to {out}");
    for (position, (_, ms)) in order.iter().enumerate().take(10) {
        println!("  Спикер {:<3} {:.0} min of speech", position + 1, *ms as f32 / 60_000.0);
    }
}
