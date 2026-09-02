//! Which of the segments the voice-activity detector hands over hold speech?
//!
//! `docs/transcription.md` records the failure this exists to fix: over a real
//! meeting the detector opened segments that held breath and room tone, and an
//! empty transcript was the only thing catching them. When levelling made the
//! recogniser talkative, thirteen of those came back as "Thank you." and
//! "Yeah." in a meeting held in Russian.
//!
//! Loudness cannot be the test. The same document records why: measured per
//! utterance, this speaker's real speech sat at 0.10 and 0.16 RMS while
//! everything else sat between 0.00006 and 0.008 — but a genuinely faint talker
//! would sit down there too, and silencing them is a worse failure than a
//! stray "Yeah."
//!
//! So this prints a level measure beside two that do not depend on level, over
//! the segments the detector actually produced, and lets the numbers choose.
//!
//! Given a recogniser as a third argument it also decodes each segment, so the
//! features and the label come from the same audio and no timestamps have to be
//! matched up afterwards.
//!
//!     cargo run --release -p shi-pipeline --example speech_gate -- \
//!         target/dev-bundles/meeting_capture.log.mic.f32 models/silero_vad.onnx \
//!         models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8

use std::sync::Arc;

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use shi_pipeline::{ModelPaths, Transcriber, load_recognizer};

const RATE: u32 = 16_000;
const VAD_BUFFER_SECONDS: f32 = 60.0;

/// Frame length for the pitch search: long enough to hold two periods of the
/// lowest voice this looks for.
const FRAME: usize = 480; // 30 ms
const HOP: usize = 160; // 10 ms

/// A human voice is periodic somewhere in here. Below is a hum, above is a
/// whistle, and neither is what a meeting is made of.
const LOWEST_HZ: f32 = 70.0;
const HIGHEST_HZ: f32 = 400.0;

/// How strongly a frame must repeat itself to count as voiced.
const PERIODIC: f32 = 0.5;

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Peak over RMS. Speech is spiky; a fan is not.
fn crest(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    let level = rms(samples);
    if level <= f32::EPSILON { 0.0 } else { peak / level }
}

/// The share of frames that carry a pitch.
///
/// A voice repeats itself at its own fundamental; breath, room tone and a fan
/// repeat themselves at no lag at all. Normalising the correlation by the
/// energy of both windows makes the answer independent of how loud any of it
/// is, which is the property the level itself does not have.
fn voiced_fraction(samples: &[f32]) -> (f32, u32) {
    let shortest = (RATE as f32 / HIGHEST_HZ) as usize;
    let longest = (RATE as f32 / LOWEST_HZ) as usize;
    if samples.len() < FRAME + longest {
        return (0.0, 0);
    }

    let mut frames = 0usize;
    let mut voiced = 0usize;
    let mut run = 0usize;
    let mut longest_run = 0usize;
    let mut at = 0;
    while at + FRAME + longest <= samples.len() {
        let window = &samples[at..at + FRAME];
        let mean = window.iter().sum::<f32>() / FRAME as f32;

        let mut best: f32 = 0.0;
        for lag in shortest..=longest {
            let shifted = &samples[at + lag..at + lag + FRAME];
            let (mut dot, mut left, mut right) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..FRAME {
                let a = window[i] - mean;
                let b = shifted[i] - mean;
                dot += a * b;
                left += a * a;
                right += b * b;
            }
            let norm = (left * right).sqrt();
            if norm > f32::EPSILON {
                best = best.max(dot / norm);
            }
        }

        frames += 1;
        if best >= PERIODIC {
            voiced += 1;
            run += 1;
            longest_run = longest_run.max(run);
        } else {
            run = 0;
        }
        at += HOP;
    }

    let fraction = if frames == 0 { 0.0 } else { voiced as f32 / frames as f32 };
    // The longest unbroken stretch of voicing, in milliseconds. A fraction is
    // diluted by whatever silence the segment happens to contain; a run is not.
    let run_ms = longest_run * HOP * 1000 / 16_000;
    (fraction, run_ms as u32)
}

fn read_dump(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dump = args.next().expect("usage: speech_gate <stream.f32> <silero.onnx>");
    let silero = args.next().expect("path to silero_vad.onnx");
    let recogniser: Option<Arc<dyn Transcriber>> = args.next().map(|dir| {
        load_recognizer(&ModelPaths::nemo_transducer(std::path::Path::new(&dir)), 4)
            .expect("load the recogniser")
    });

    let samples = read_dump(&dump);
    eprintln!(
        "{:.0} s at {RATE} Hz from {dump}",
        samples.len() as f32 / RATE as f32
    );

    let mut config = VadModelConfig::default();
    // The pipeline's own defaults unless the environment says otherwise. The
    // detector's threshold was never swept on real audio: 0.5 came from the
    // plan, and the segments it opened over an empty room are what the voice
    // gate exists to catch. Whether a higher bar catches them first — and what
    // it costs the stream that carries the conversation — is measured here.
    let env_or = |name: &str, fallback: f32| -> f32 {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
    };
    let threshold = env_or("VAD_THRESHOLD", 0.5);
    let min_speech = env_or("VAD_MIN_SPEECH_MS", 250.0) / 1000.0;
    let min_silence = env_or("VAD_MIN_SILENCE_MS", 500.0) / 1000.0;
    eprintln!("vad threshold {threshold}, min speech {min_speech}s, min silence {min_silence}s");
    config.silero_vad = SileroVadModelConfig {
        model: Some(silero.clone()),
        threshold,
        min_silence_duration: min_silence,
        min_speech_duration: min_speech,
        window_size: 512,
        max_speech_duration: 20.0,
    };
    config.sample_rate = RATE as i32;

    let mut vad = VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
        .expect("create the voice activity detector");

    println!("start_ms\tms\trms\tcrest\tvoiced\trun_ms\ttext");

    let mut report = |vad: &mut VoiceActivityDetector| {
        // `recogniser` is borrowed here rather than moved: the closure runs
        // once per block and once more at the flush.
        while let Some(segment) = vad.front() {
            let start = segment.start().max(0) as u64;
            let audio = segment.samples().to_vec();
            vad.pop();
            drop(segment);

            let start_ms = start * 1000 / RATE as u64;
            let length_ms = audio.len() as u64 * 1000 / RATE as u64;
            let text = recogniser
                .as_ref()
                .and_then(|r| r.transcribe(&audio).ok())
                .map(|t| t.text.replace(['\t', '\n'], " "))
                .unwrap_or_default();
            let (fraction, run_ms) = voiced_fraction(&audio);
            println!(
                "{start_ms}\t{length_ms}\t{:.5}\t{:.2}\t{fraction:.3}\t{run_ms}\t{text}",
                rms(&audio),
                crest(&audio),
            );
        }
    };

    for block in samples.chunks(RATE as usize) {
        vad.accept_waveform(block);
        report(&mut vad);
    }
    vad.flush();
    report(&mut vad);
}
