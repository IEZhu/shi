//! Run several recognisers over the same audio and print what each one heard.
//!
//!     cargo run --release -p shi-pipeline --example compare_models -- \
//!         bench.wav out.md models/dir-a models/dir-b ...
//!
//! There is no reference transcript for a real meeting, so this does not score
//! anything. It puts the same utterances side by side and reports the cost, and
//! leaves the judgement to somebody who was in the room.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shi_audio::StreamKind;
use shi_pipeline::{
    ModelPaths, PipelineEvent, SherpaTranscriber, StreamPipeline, Transcriber, VadSettings,
};

/// Work out how a model directory is laid out, rather than being told.
fn detect(dir: &Path) -> Option<ModelPaths> {
    let has = |name: &str| dir.join(name).is_file();

    // Each file is resolved on its own: GigaAM ships a quantised encoder beside
    // a full-precision decoder and joiner, so requiring one suffix throughout
    // would reject a model that works perfectly well.
    let part = |stem: &str| {
        [".int8.onnx", ".onnx"]
            .iter()
            .map(|suffix| dir.join(format!("{stem}{suffix}")))
            .find(|path| path.is_file())
    };
    if let (Some(encoder), Some(decoder), Some(joiner)) =
        (part("encoder"), part("decoder"), part("joiner"))
        && has("tokens.txt")
    {
        return Some(ModelPaths::NemoTransducer {
            encoder,
            decoder,
            joiner,
            tokens: dir.join("tokens.txt"),
        });
    }

    // Whisper prefixes every file with the model size: `large-v3-encoder.onnx`.
    let entry = std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        name.strip_suffix("-tokens.txt").map(str::to_string)
    })?;
    for suffix in [".int8.onnx", ".onnx"] {
        if has(&format!("{entry}-encoder{suffix}")) && has(&format!("{entry}-decoder{suffix}")) {
            return Some(ModelPaths::Whisper {
                encoder: dir.join(format!("{entry}-encoder{suffix}")),
                decoder: dir.join(format!("{entry}-decoder{suffix}")),
                tokens: dir.join(format!("{entry}-tokens.txt")),
                // Left to detect: fixing it would decide the very question
                // this comparison is asking.
                language: None,
            });
        }
    }
    None
}

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open benchmark audio");
    let rate = reader.spec().sample_rate;
    let samples = reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect();
    (samples, rate)
}

fn stamp(seconds: f32) -> String {
    format!("{:02}:{:05.2}", (seconds as u32) / 60, seconds % 60.0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = args.next().expect("usage: compare_models <wav> <out.md> <dir>...");
    let out = args.next().expect("output path");
    let dirs: Vec<PathBuf> = args.map(PathBuf::from).collect();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let silero = format!("{root}/models/silero_vad.onnx");

    let (audio, rate) = read_wav(&wav);
    let seconds = audio.len() as f32 / rate as f32;
    println!("benchmark: {seconds:.1}s at {rate} Hz across {} models\n", dirs.len());

    let mut report = format!(
        "# Сравнение моделей распознавания\n\nОтрезок: {seconds:.0} с реальной встречи, \
         русская речь с английскими техническими терминами.\n"
    );

    for dir in &dirs {
        let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let Some(paths) = detect(dir) else {
            println!("{name}: layout not recognised, skipped");
            report.push_str(&format!("\n## {name}\n\nНе удалось определить раскладку файлов.\n"));
            continue;
        };

        let loading = Instant::now();
        let transcriber: Arc<dyn Transcriber> = match SherpaTranscriber::load(&paths, 4) {
            Ok(t) => Arc::new(t),
            Err(err) => {
                println!("{name}: would not load: {err}");
                report.push_str(&format!("\n## {name}\n\nНе загрузилась: {err}\n"));
                continue;
            }
        };
        let load_time = loading.elapsed();

        let mut pipeline =
            StreamPipeline::new(StreamKind::System, rate, &silero, transcriber, VadSettings::default())
                .expect("build pipeline");

        let began = Instant::now();
        let mut lines = Vec::new();
        for block in audio.chunks(rate as usize / 2) {
            // A zero clock means the pipeline never decides it is behind and
            // never pads with silence: the file is contiguous, and its own
            // sample count is the only timeline there is.
            for event in pipeline.push_at(block, std::time::Duration::ZERO) {
                if let PipelineEvent::Final { start, text, .. } = event
                    && !text.trim().is_empty()
                {
                    lines.push((start.as_secs_f32(), text));
                }
            }
        }
        for event in pipeline.flush() {
            if let PipelineEvent::Final { start, text, .. } = event
                && !text.trim().is_empty()
            {
                lines.push((start.as_secs_f32(), text));
            }
        }
        let spent = began.elapsed();

        let chars: usize = lines.iter().map(|(_, t)| t.chars().count()).sum();
        println!(
            "{name}\n  loaded in {:.1}s, decoded {seconds:.0}s in {:.1}s ({:.1}x realtime), \
             {} utterances, {chars} characters",
            load_time.as_secs_f32(),
            spent.as_secs_f32(),
            seconds / spent.as_secs_f32(),
            lines.len()
        );

        report.push_str(&format!(
            "\n## {name}\n\n{:.1}× быстрее реального времени · {} реплик · {chars} символов\n\n",
            seconds / spent.as_secs_f32(),
            lines.len()
        ));
        for (at, text) in &lines {
            report.push_str(&format!("**[{}]** {text}\n\n", stamp(*at)));
        }
    }

    std::fs::write(&out, report).expect("write report");
    println!("\nwritten to {out}");
}
