//! Measure echo suppression on real acoustics.
//!
//! The unit tests simulate the loudspeaker path. This one uses it: play speech
//! through the speakers with nobody talking, and every microphone utterance is
//! by definition an echo. Anything the mic transcribes is a duplicated line.
//!
//!     PKG=shi-pipeline scripts/dev-run.sh echo_probe 20
//!
//! Reports each utterance with the score that decided it, so a miss can be
//! read off rather than guessed at.

use std::sync::Arc;
use std::time::{Duration, Instant};

use shi_audio::{AudioSource, MicSource, StreamHandle, StreamKind, SystemSource};
use shi_pipeline::{
    EchoReference, PipelineEvent, StreamPipeline, Transcriber, Transcript, VadSettings,
};

/// Stand-in for the recogniser: this measures the echo verdict, not the words.
struct Counted;
impl Transcriber for Counted {
    fn transcribe(&self, samples: &[f32]) -> shi_pipeline::Result<Transcript> {
        Ok(Transcript {
            text: format!("{:.2}s of speech", samples.len() as f32 / 16_000.0),
            tokens: Vec::new(),
            token_offsets: Vec::new(),
        })
    }
    fn model_id(&self) -> &str {
        "counted"
    }
}

fn redirect_stdout(path: &str) {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let Ok(file) = std::fs::File::create(path) else {
        return;
    };
    let fd = file.into_raw_fd();
    // SAFETY: `fd` is freshly opened, and dup2 onto stdout is how it is done.
    unsafe {
        libc::dup2(fd, std::io::stdout().as_raw_fd());
        libc::close(fd);
    }
}

fn drain(handle: &mut StreamHandle, buffer: &mut Vec<f32>) -> usize {
    buffer.clear();
    while let Ok(sample) = handle.consumer.pop() {
        buffer.push(sample);
    }
    buffer.len()
}

fn main() {
    if let Ok(path) = std::env::var("DEV_LOG") {
        redirect_stdout(&path);
    }
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .without_time()
        .init();

    let seconds: u64 = std::env::var("PROBE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let silero = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/silero_vad.onnx");

    let mut mic_source = MicSource::default_device();
    let mut system_source = SystemSource::new();
    let mut mic_handle = match mic_source.start() {
        Ok(h) => h,
        Err(err) => return println!("microphone did not start: {err}"),
    };
    let mut system_handle = match system_source.start() {
        Ok(h) => h,
        Err(err) => return println!("system tap did not start: {err}"),
    };
    println!(
        "mic    {} @ {} Hz\nsystem {} @ {} Hz\n",
        mic_handle.info.device_name,
        mic_handle.info.sample_rate,
        system_handle.info.device_name,
        system_handle.info.sample_rate
    );

    let reference = Arc::new(EchoReference::default());
    let build = |kind: StreamKind, rate: u32| {
        StreamPipeline::new(kind, rate, silero, Arc::new(Counted), VadSettings::default())
            .expect("build pipeline")
    };
    let mut system = build(StreamKind::System, system_handle.info.sample_rate);
    system.publish_echo_reference(Arc::clone(&reference));
    let mut mic = build(StreamKind::Mic, mic_handle.info.sample_rate);
    mic.suppress_echo_of(Arc::clone(&reference));

    // Same as the app: the rings filled while the pipelines were being built,
    // and keeping that would put the two streams on different clocks.
    println!(
        "discarded backlog: mic {} samples, system {} samples\n",
        mic_handle.discard_backlog(),
        system_handle.discard_backlog()
    );

    let origin = Instant::now();
    system.started_at(origin);
    mic.started_at(origin);

    let mut buffer = Vec::new();
    let (mut mic_finals, mut system_finals) = (0u32, 0u32);

    while origin.elapsed() < Duration::from_secs(seconds) {
        // System first: the microphone must never be judged against a
        // reference that has not yet heard the moment being judged.
        if drain(&mut system_handle, &mut buffer) > 0 {
            for event in system.push_at(&buffer, origin.elapsed()) {
                if let PipelineEvent::Final { start, text, .. } = &event {
                    system_finals += 1;
                    println!("system {:>7.2}s  {text}", start.as_secs_f32());
                }
            }
        }
        if drain(&mut mic_handle, &mut buffer) > 0 {
            for event in mic.push_at(&buffer, origin.elapsed()) {
                if let PipelineEvent::Final { start, text, .. } = &event {
                    mic_finals += 1;
                    println!("MIC    {:>7.2}s  {text}   <- reached the transcript", start.as_secs_f32());
                }
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    system.flush();
    mic.flush();
    mic_source.stop();
    system_source.stop();

    println!(
        "\nsystem utterances {system_finals}\nmic utterances    {mic_finals}\nsuppressed        {}",
        mic.suppressed_echo()
    );
    println!(
        "\nnobody was speaking, so every microphone utterance is an echo:\n  {} ",
        if mic_finals == 0 { "PASS — none survived" } else { "FAIL — lines will be duplicated" }
    );
}
