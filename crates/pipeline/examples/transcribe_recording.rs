//! Transcribe a recording made by the `meeting_capture` example.
//!
//!     cargo run --release -p shi-pipeline --example transcribe_recording -- \
//!         target/dev-bundles/meeting_capture.log meeting.md
//!
//! Recording during a call and recognising afterwards keeps the CPU out of the
//! meeting. Both streams go through the production pipeline: the same VAD, the
//! same recogniser, the same diarizer, and the same echo suppression — fed in
//! lockstep so the reference is always ahead of the microphone being judged.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use shi_audio::StreamKind;
use shi_pipeline::{
    Attribution, AudioTap, DEFAULT_SESSION_THRESHOLD, EchoReference, ModelPaths, PipelineEvent,
    SherpaTranscriber, SpeakerTracker, Span, StreamPipeline, Thresholds, Transcriber, VadSettings,
    rediarize,
};

/// How much audio to hand the pipelines per turn. One second keeps the two
/// streams interleaved closely enough that the echo reference always covers
/// the microphone audio being judged against it.
const STEP: f32 = 1.0;

/// Keeps the system stream's resampled audio, which is what the utterance
/// timestamps refer to — the offline clustering has to slice the very same
/// timeline the online pass produced, inserted silence and all.
#[derive(Clone, Default)]
struct Kept(Arc<std::sync::Mutex<Vec<f32>>>);

impl AudioTap for Kept {
    fn write(&mut self, samples: &[f32]) {
        if let Ok(mut held) = self.0.lock() {
            held.extend_from_slice(samples);
        }
    }
    fn finish(self: Box<Self>) {}
}

struct Raw {
    reader: BufReader<File>,
    rate: u32,
    pushed: u64,
}

impl Raw {
    fn open(base: &str, label: &str) -> std::io::Result<Self> {
        let path = format!("{base}.{label}.f32");
        let meta = std::fs::read_to_string(format!("{path}.meta")).unwrap_or_default();
        let rate = meta
            .split_whitespace()
            .find_map(|f| f.strip_prefix("rate=")?.parse().ok())
            .unwrap_or(16_000);
        Ok(Self {
            reader: BufReader::with_capacity(1 << 20, File::open(&path)?),
            rate,
            pushed: 0,
        })
    }

    fn next_step(&mut self) -> Vec<f32> {
        let wanted = (self.rate as f32 * STEP) as usize;
        let mut bytes = vec![0u8; wanted * 4];
        let mut filled = 0;
        while filled < bytes.len() {
            match self.reader.read(&mut bytes[filled..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => filled += n,
            }
        }
        let samples: Vec<f32> = bytes[..filled - filled % 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        self.pushed += samples.len() as u64;
        samples
    }

    fn elapsed(&self) -> Duration {
        Duration::from_secs_f64(self.pushed as f64 / self.rate as f64)
    }
}

fn stamp(at: Duration) -> String {
    let total = at.as_secs();
    format!("{:02}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60)
}

fn who(stream: StreamKind, speaker: &Option<Attribution>) -> String {
    if stream == StreamKind::Mic {
        return "Вы".into();
    }
    match speaker {
        Some(Attribution::Known { name, .. }) => name.clone(),
        Some(Attribution::Slot { id, .. }) | Some(Attribution::Continuation { id }) => {
            format!("Спикер {id}")
        }
        _ => "Собеседник".into(),
    }
}

/// Save what the expensive pass produced, so speaker labelling can be tried
/// again in seconds instead of another hour of recognition.
fn dump_state(base: &str, kept: &Kept, lines: &[(Duration, String, String, Option<Span>)]) {
    if let Ok(audio) = kept.0.lock() {
        let mut bytes = Vec::with_capacity(audio.len() * 4);
        for sample in audio.iter() {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let _ = std::fs::write(format!("{base}.system16k.f32"), bytes);
    }
    let mut tsv = String::from("id\tstart_ms\tend_ms\tstream\ttext\n");
    for (at, who, text, span) in lines {
        let (id, end) = span.map_or((-1, 0), |s| (s.id, s.end_ms));
        tsv.push_str(&format!(
            "{id}\t{}\t{end}\t{who}\t{}\n",
            at.as_millis(),
            text.replace('\t', " ").replace('\n', " ")
        ));
    }
    let _ = std::fs::write(format!("{base}.spans.tsv"), tsv);
}

fn distinct_speakers(lines: &[(Duration, String, String, Option<Span>)]) -> usize {
    lines
        .iter()
        .map(|(_, who, _, _)| who.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// Re-label the system stream's utterances from the finished recording.
fn regroup_speakers(
    lines: &mut [(Duration, String, String, Option<Span>)],
    kept: &Kept,
    root: &str,
) {
    let Ok(tracker) = SpeakerTracker::new(
        &format!("{root}/models/nemo_en_titanet_small.onnx"),
        4,
        Thresholds::default(),
    ) else {
        eprintln!("offline clustering skipped: the speaker model would not load");
        return;
    };
    let Ok(audio) = kept.0.lock() else { return };
    let spans: Vec<Span> = lines.iter().filter_map(|(_, _, _, span)| span.clone()).collect();
    if spans.is_empty() {
        return;
    }

    let assigned: std::collections::BTreeMap<i64, u32> =
        rediarize(&tracker, &audio, &spans, DEFAULT_SESSION_THRESHOLD)
            .into_iter()
            .collect();

    // Everything the clustering judged gets its new label. Everything it
    // skipped — utterances too short to carry a voice — must NOT keep the
    // online label: that would leave two numbering schemes side by side and
    // inflate the count rather than reduce it, which is exactly what happened
    // the first time (39 speakers became 53). A short interjection belongs to
    // whoever was speaking around it.
    let mut carried: Option<u32> = None;
    for (_, who, _, span) in lines.iter_mut() {
        let Some(span) = span else { continue };
        match assigned.get(&span.id) {
            Some(slot) => {
                carried = Some(*slot);
                *who = format!("Спикер {slot}");
            }
            None => {
                if let Some(slot) = carried {
                    *who = format!("Спикер {slot}");
                }
            }
        }
    }
}

fn main() {
    // Every segment the voice gate rejects is logged at info level with the
    // numbers that decided it. Without a subscriber those lines go nowhere,
    // which is how a probe once ran for ten minutes and reported nothing.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    let mut args = std::env::args().skip(1);
    let base = args.next().expect("usage: transcribe_recording <base> <out.md>");
    let out = args.next().unwrap_or_else(|| "meeting.md".into());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    let mut mic_raw = Raw::open(&base, "mic").expect("open microphone recording");
    let mut sys_raw = Raw::open(&base, "system").expect("open system recording");

    eprintln!("loading the recogniser...");
    let transcriber: Arc<dyn Transcriber> = Arc::new(
        SherpaTranscriber::load(
            &ModelPaths::parakeet_int8(std::path::PathBuf::from(format!(
                "{root}/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
            ))),
            4,
        )
        .expect("load recogniser"),
    );
    // SHI_SILERO points at another voice-activity model. The 2025-07 export in
    // `models/` behaves exactly like Silero v4; v5 opened six microphone
    // segments where v4 opened 277 over the same meeting, and the comparison
    // has to run through this pipeline rather than a stand-in.
    let silero = std::env::var("SHI_SILERO")
        .unwrap_or_else(|_| format!("{root}/models/silero_vad.onnx"));
    eprintln!("voice activity model: {silero}");

    // SHI_MIN_VOICED_MS=0 turns the voice gate off, which is how the before and
    // after in docs/transcription.md were measured against each other.
    let mut vad = VadSettings::default();
    if let Ok(value) = std::env::var("SHI_MIN_VOICED_MS")
        && let Ok(parsed) = value.parse::<u32>()
    {
        eprintln!("voice gate at {parsed} ms");
        vad.min_voiced_ms = parsed;
    }
    // SHI_LONGEST_DECODE_MS caps what one decode may cover. The corpus says
    // shorter is more accurate; the meeting has no reference, so the two
    // transcripts are for a person who was there to compare.
    if let Ok(value) = std::env::var("SHI_LONGEST_DECODE_MS")
        && let Ok(parsed) = value.parse::<u64>()
    {
        eprintln!("longest decode {parsed} ms");
        vad.longest_decode = Duration::from_millis(parsed);
    }

    let build = |kind: StreamKind, rate: u32| {
        StreamPipeline::new(kind, rate, &silero, Arc::clone(&transcriber), vad)
            .expect("build pipeline")
    };
    let mut system = build(StreamKind::System, sys_raw.rate);
    let mut mic = build(StreamKind::Mic, mic_raw.rate);

    let kept = Kept::default();
    system.record_to(Box::new(kept.clone()));

    let echo = Arc::new(EchoReference::default());
    system.publish_echo_reference(Arc::clone(&echo));
    mic.suppress_echo_of(echo);

    match SpeakerTracker::new(
        &format!("{root}/models/nemo_en_titanet_small.onnx"),
        4,
        Thresholds::default(),
    ) {
        Ok(tracker) => system.identify_speakers(tracker),
        Err(err) => eprintln!("speaker tracking unavailable: {err}"),
    }

    // start, speaker label, text, and — for the system stream — the span the
    // offline clustering will re-judge.
    let mut lines: Vec<(Duration, String, String, Option<Span>)> = Vec::new();
    let mut collect = |events: Vec<PipelineEvent>,
                       into: &mut Vec<(Duration, String, String, Option<Span>)>| {
        for event in events {
            if let PipelineEvent::Final { stream, start, end, text, speaker, .. } = event
                && !text.trim().is_empty()
            {
                let span = (stream == StreamKind::System).then(|| Span {
                    id: into.len() as i64,
                    start_ms: start.as_millis() as i64,
                    end_ms: end.as_millis() as i64,
                });
                into.push((start, who(stream, &speaker), text, span));
            }
        }
    };

    let began = Instant::now();
    let mut step = 0u64;
    loop {
        // System first: the microphone must never be judged against a
        // reference that has not yet heard the moment in question.
        let sys_block = sys_raw.next_step();
        let mic_block = mic_raw.next_step();
        if sys_block.is_empty() && mic_block.is_empty() {
            break;
        }
        if !sys_block.is_empty() {
            let at = sys_raw.elapsed();
            collect(system.push_at(&sys_block, at), &mut lines);
        }
        if !mic_block.is_empty() {
            let at = mic_raw.elapsed();
            collect(mic.push_at(&mic_block, at), &mut lines);
        }

        step += 1;
        if step % 60 == 0 {
            eprintln!(
                "  {} of audio done in {:.0}s ({:.1}x realtime), {} lines",
                stamp(sys_raw.elapsed()),
                began.elapsed().as_secs_f32(),
                sys_raw.elapsed().as_secs_f32() / began.elapsed().as_secs_f32().max(0.001),
                lines.len()
            );
        }
    }
    collect(system.flush(), &mut lines);
    collect(mic.flush(), &mut lines);

    // The online tracker only ever has the past to go on, so one voice can open
    // a slot that later evidence would have merged — 38 speakers for a meeting
    // that had a handful. With the whole recording in hand every utterance can
    // be compared with every other, which is why keeping the audio is worth the
    // disk.
    dump_state(&base, &kept, &lines);

    let before = distinct_speakers(&lines);
    regroup_speakers(&mut lines, &kept, root);
    eprintln!(
        "speakers: {before} online -> {} after offline clustering",
        distinct_speakers(&lines)
    );

    lines.sort_by_key(|(at, _, _, _)| *at);

    let mut file = File::create(&out).expect("create transcript");
    let duration = sys_raw.elapsed().max(mic_raw.elapsed());
    writeln!(
        file,
        "---\ntitle: Встреча\nduration: {}\nsuppressed_echo: {}\n---\n",
        stamp(duration),
        mic.suppressed_echo()
    )
    .ok();

    let mut previous = String::new();
    for (at, speaker, text, _) in &lines {
        if *speaker != previous {
            writeln!(file, "\n**[{}] {}:** {}", stamp(*at), speaker, text).ok();
            previous = speaker.clone();
        } else {
            writeln!(file, "{text}").ok();
        }
    }

    eprintln!(
        "\n{} lines, {} suppressed as echo, written to {out}",
        lines.len(),
        mic.suppressed_echo()
    );
}
