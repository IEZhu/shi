use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;
use shi_audio::{StreamHandle, StreamKind};
use shi_pipeline::{
    PipelineEvent, SherpaTranscriber, StreamPipeline, Transcriber, VadSettings,
};
use shi_store::{NewSegment, Store, markdown};
use tauri::{AppHandle, Emitter};

use crate::capture::{Capture, Readiness, StreamStatus, StartedStreams, READINESS_EVENT};
use crate::config::Config;
use crate::error::AppError;

/// Event name carrying [`TranscriptEvent`] to the frontend.
pub const TRANSCRIPT_EVENT: &str = "transcript";

/// Readiness meter refresh rate.
const METER_TICK: Duration = Duration::from_millis(50);
/// How long a pipeline worker waits when its ring is empty.
const IDLE_SLEEP: Duration = Duration::from_millis(10);
/// Samples pulled from a ring per turn, roughly 20 ms at 48 kHz.
const READ_CHUNK: usize = 1024;
/// Markdown is re-rendered at most this often while a meeting runs; the
/// database already holds every finalised utterance, so this is only about
/// keeping the file fresh for anyone watching it.
const RENDER_INTERVAL: Duration = Duration::from_secs(10);

/// What the transcript view receives.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum TranscriptEvent {
    /// Provisional text for an utterance still being spoken. Always replaced.
    Draft {
        stream: String,
        start_ms: i64,
        text: String,
    },
    /// A completed utterance. This one is persisted.
    Final {
        id: i64,
        stream: String,
        start_ms: i64,
        end_ms: i64,
        speaker: Option<String>,
        text: String,
    },
    /// An utterance ended without text, so drop the draft on screen.
    DraftAbandoned { stream: String },
}

/// A meeting in progress, or the absence of one.
pub struct Session {
    config: Config,
    capture: Capture,
    store: Arc<Mutex<Store>>,
    state: Arc<Mutex<SessionState>>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct SessionState {
    readiness: Readiness,
    meeting_id: Option<i64>,
}

/// Shared measurement so the readiness panel can show what decoding costs.
type SharedRtf = Arc<Mutex<Option<f32>>>;

impl Session {
    pub fn new(config: Config) -> Result<Self, AppError> {
        config.ensure_dirs()?;
        let store = Store::open(config.database())?;
        let readiness = Readiness::idle(config.missing_models());

        Ok(Self {
            config,
            capture: Capture::new(),
            store: Arc::new(Mutex::new(store)),
            state: Arc::new(Mutex::new(SessionState {
                readiness,
                meeting_id: None,
            })),
            stop: Arc::new(AtomicBool::new(false)),
            workers: Vec::new(),
        })
    }

    pub fn readiness(&self) -> Readiness {
        lock(&self.state).readiness.clone()
    }

    pub fn is_running(&self) -> bool {
        !self.workers.is_empty()
    }

    /// Start capture without transcribing — the pre-meeting check.
    pub fn start_check(&mut self, app: AppHandle) -> Result<Readiness, AppError> {
        self.start(app, None)
    }

    /// Start capture and transcribe into a new meeting.
    pub fn start_meeting(&mut self, app: AppHandle, title: &str) -> Result<Readiness, AppError> {
        let missing = self.config.missing_models();
        if !missing.is_empty() {
            return Err(AppError::ModelsMissing(missing));
        }
        self.start(app, Some(title.to_string()))
    }

    fn start(&mut self, app: AppHandle, title: Option<String>) -> Result<Readiness, AppError> {
        if self.is_running() {
            return Ok(self.readiness());
        }

        let streams = self.capture.start();
        if !streams.any_running() {
            let readiness = self.compose_idle_readiness(&streams);
            self.set_readiness(readiness.clone());
            self.capture.stop();
            return Ok(readiness);
        }

        let meeting_id = match &title {
            Some(title) => {
                let started_at = jiff::Zoned::now().to_string();
                let model_id = self
                    .config
                    .recognizer_dir()
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".into());
                let meeting =
                    lock(&self.store).start_meeting(title, &started_at, &model_id)?;
                Some(meeting.id)
            }
            None => None,
        };

        self.stop.store(false, Ordering::SeqCst);
        let rtf: SharedRtf = Arc::new(Mutex::new(None));

        let StartedStreams { mic, system } = streams;
        let mut meters = Vec::new();

        for handle in [mic, system] {
            match handle {
                Ok(handle) => {
                    let info = handle.info.clone();
                    let stats = Arc::clone(&handle.stats);
                    meters.push(Ok((info.clone(), stats)));

                    self.workers.push(self.spawn_stream_worker(
                        app.clone(),
                        handle,
                        meeting_id,
                        Arc::clone(&rtf),
                    )?);
                }
                Err(status) => meters.push(Err(status)),
            }
        }

        self.workers.push(self.spawn_meter(app, meters, meeting_id.is_some(), rtf));
        lock(&self.state).meeting_id = meeting_id;

        Ok(self.readiness())
    }

    /// Drain one capture ring into its pipeline, forwarding and storing results.
    fn spawn_stream_worker(
        &self,
        app: AppHandle,
        mut handle: StreamHandle,
        meeting_id: Option<i64>,
        rtf: SharedRtf,
    ) -> Result<JoinHandle<()>, AppError> {
        let kind = handle.info.kind;
        let stop = Arc::clone(&self.stop);
        let store = Arc::clone(&self.store);
        let config = self.config.clone();

        // Only build the heavy machinery when there is a meeting to transcribe;
        // a readiness check should not spend 600 MB and a model load.
        let pipeline = match meeting_id {
            Some(_) => {
                let transcriber: Arc<dyn Transcriber> = Arc::new(SherpaTranscriber::load(
                    &config.recognizer(),
                    config.asr_threads,
                )?);
                Some(StreamPipeline::new(
                    kind,
                    handle.info.sample_rate,
                    &config.silero().to_string_lossy(),
                    transcriber,
                    VadSettings::default(),
                )?)
            }
            None => None,
        };

        Ok(thread::spawn(move || {
            let mut pipeline = pipeline;
            let mut buffer = vec![0.0f32; READ_CHUNK];
            let mut last_render = Instant::now();

            while !stop.load(Ordering::SeqCst) {
                let mut filled = 0;
                while filled < buffer.len() {
                    match handle.consumer.pop() {
                        Ok(sample) => {
                            buffer[filled] = sample;
                            filled += 1;
                        }
                        Err(_) => break,
                    }
                }

                if filled == 0 {
                    thread::sleep(IDLE_SLEEP);
                    continue;
                }

                let Some(pipeline) = pipeline.as_mut() else {
                    // Readiness check: the ring still has to be drained or the
                    // drop counter climbs and the meters start lying.
                    continue;
                };

                let events = pipeline.push(&buffer[..filled]);
                if let Ok(mut slot) = rtf.lock() {
                    *slot = Some(pipeline.rtf());
                }
                emit_events(&app, &store, &config, meeting_id, events);

                if last_render.elapsed() >= RENDER_INTERVAL {
                    last_render = Instant::now();
                    if let Some(id) = meeting_id {
                        render_markdown(&store, &config, id);
                    }
                }
            }

            // Closing time: whatever speech is still buffered is still speech.
            if let Some(pipeline) = pipeline.as_mut() {
                let events = pipeline.flush();
                emit_events(&app, &store, &config, meeting_id, events);
            }
        }))
    }

    /// Publish capture health at a steady rate, independent of audio activity.
    fn spawn_meter(
        &self,
        app: AppHandle,
        meters: Vec<Result<(shi_audio::SourceInfo, Arc<shi_audio::StreamStats>), StreamStatus>>,
        recording: bool,
        rtf: SharedRtf,
    ) -> JoinHandle<()> {
        let stop = Arc::clone(&self.stop);
        let state = Arc::clone(&self.state);
        let missing = self.config.missing_models();

        thread::spawn(move || {
            let mut tick: u32 = 0;
            while !stop.load(Ordering::SeqCst) {
                thread::sleep(METER_TICK);

                let mut statuses = meters.iter().map(|meter| match meter {
                    Ok((info, stats)) => StreamStatus::sample(info, stats),
                    Err(status) => status.clone(),
                });
                let mic = statuses.next().unwrap_or(StreamStatus::stopped(StreamKind::Mic));
                let system = statuses
                    .next()
                    .unwrap_or(StreamStatus::stopped(StreamKind::System));

                let readiness = Readiness {
                    mic,
                    system,
                    missing_models: missing.clone(),
                    recording,
                    rtf: rtf.lock().ok().and_then(|r| *r),
                };

                // Once a second, leave something diagnosable behind. Capture
                // problems get reported as "it recorded nothing", and this is
                // what turns that into an answer.
                tick += 1;
                if tick % 20 == 0 {
                    tracing::info!(
                        mic = ?readiness.mic.verdict,
                        mic_dropped = readiness.mic.frames_dropped,
                        system = ?readiness.system.verdict,
                        system_dropped = readiness.system.frames_dropped,
                        rtf = ?readiness.rtf,
                        "readiness"
                    );
                }

                if let Ok(mut slot) = state.lock() {
                    slot.readiness = readiness.clone();
                }
                if app.emit(READINESS_EVENT, &readiness).is_err() {
                    break;
                }
            }
        })
    }

    pub fn stop(&mut self) -> Readiness {
        self.stop.store(true, Ordering::SeqCst);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        self.capture.stop();

        let meeting_id = lock(&self.state).meeting_id.take();
        if let Some(id) = meeting_id {
            let ended_at = jiff::Zoned::now().to_string();
            if let Err(err) = lock(&self.store).finish_meeting(id, &ended_at) {
                tracing::error!("cannot close meeting {id}: {err}");
            }
            render_markdown(&self.store, &self.config, id);
        }

        let readiness = Readiness::idle(self.config.missing_models());
        self.set_readiness(readiness.clone());
        readiness
    }

    pub fn store(&self) -> Arc<Mutex<Store>> {
        Arc::clone(&self.store)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn compose_idle_readiness(&self, streams: &StartedStreams) -> Readiness {
        let mut readiness = Readiness::idle(self.config.missing_models());
        if let Err(status) = &streams.mic {
            readiness.mic = status.clone();
        }
        if let Err(status) = &streams.system {
            readiness.system = status.clone();
        }
        readiness
    }

    fn set_readiness(&self, readiness: Readiness) {
        lock(&self.state).readiness = readiness;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.is_running() {
            self.stop();
        }
    }
}

/// Persist finals, then tell the UI about everything.
fn emit_events(
    app: &AppHandle,
    store: &Arc<Mutex<Store>>,
    config: &Config,
    meeting_id: Option<i64>,
    events: Vec<PipelineEvent>,
) {
    for event in events {
        let payload = match event {
            PipelineEvent::Draft {
                stream,
                start,
                text,
            } => TranscriptEvent::Draft {
                stream: stream.as_str().into(),
                start_ms: start.as_millis() as i64,
                text,
            },

            PipelineEvent::DraftAbandoned { stream } => TranscriptEvent::DraftAbandoned {
                stream: stream.as_str().into(),
            },

            PipelineEvent::Final {
                stream,
                start,
                end,
                text,
                ..
            } => {
                let segment = NewSegment {
                    stream,
                    start_ms: start.as_millis() as i64,
                    end_ms: end.as_millis() as i64,
                    // Microphone audio is the local participant by
                    // construction; the system stream waits for diarization.
                    speaker: match stream {
                        StreamKind::Mic => Some(config.markdown().unnamed_mic.clone()),
                        StreamKind::System => None,
                    },
                    text: text.clone(),
                };

                let Some(meeting_id) = meeting_id else {
                    continue;
                };
                let id = match lock(store).append_segment(meeting_id, &segment) {
                    Ok(id) => id,
                    Err(err) => {
                        tracing::error!("cannot store segment: {err}");
                        continue;
                    }
                };

                TranscriptEvent::Final {
                    id,
                    stream: stream.as_str().into(),
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    speaker: segment.speaker,
                    text,
                }
            }
        };

        if app.emit(TRANSCRIPT_EVENT, &payload).is_err() {
            return;
        }
    }
}

/// Re-render the whole file from the database. Never appended to, so a speaker
/// renamed after the fact is corrected everywhere at once.
fn render_markdown(store: &Arc<Mutex<Store>>, config: &Config, meeting_id: i64) {
    let store = lock(store);
    let (meeting, segments) = match (store.meeting(meeting_id), store.segments(meeting_id)) {
        (Ok(meeting), Ok(segments)) => (meeting, segments),
        (meeting, segments) => {
            if let Err(err) = meeting {
                tracing::error!("cannot read meeting {meeting_id}: {err}");
            }
            if let Err(err) = segments {
                tracing::error!("cannot read segments of {meeting_id}: {err}");
            }
            return;
        }
    };

    let path = config.markdown_path(&meeting);
    if let Err(err) = markdown::write_to(&path, &meeting, &segments, &config.markdown()) {
        tracing::error!("cannot write {}: {err}", path.display());
        return;
    }
    if meeting.md_path.as_deref() != Some(path.to_string_lossy().as_ref()) {
        let _ = store.set_markdown_path(meeting_id, &path.to_string_lossy());
    }
}

/// A poisoned lock means a worker panicked; the UI should keep working rather
/// than inherit the panic.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
