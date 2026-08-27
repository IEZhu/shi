use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::Serialize;
use shi_audio::{StreamHandle, StreamKind};
use shi_pipeline::{
    Attribution, AudioTap, EchoReference, PipelineEvent, SherpaTranscriber, SpeakerTracker,
    StreamPipeline, Thresholds, Transcriber, VadSettings, VoiceProfile,
};
use shi_store::{AudioStore, NewSegment, Recorder, SessionSlot, Store, markdown};
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
        /// Resolved name, when the voice is known.
        speaker: Option<String>,
        /// Unnamed voice within this meeting, shown as "Спикер N".
        slot: Option<u32>,
        text: String,
    },
    /// A voice nobody has named yet has just been heard for the first time.
    /// The UI offers to name it without interrupting the transcript.
    SpeakerDiscovered { slot: u32 },
    /// An utterance ended without text, so drop the draft on screen.
    DraftAbandoned { stream: String },
}

/// A meeting in progress, or the absence of one.
/// Lets the pipeline write recordings without knowing about files or retention.
struct RecorderTap(Recorder);

impl AudioTap for RecorderTap {
    fn write(&mut self, samples: &[f32]) {
        self.0.write(samples);
    }

    fn finish(self: Box<Self>) {
        match self.0.finish() {
            Ok(path) => tracing::info!(file = %path.display(), "recording compressed"),
            Err(err) => tracing::error!("cannot finish a recording: {err}"),
        }
    }
}

pub struct Session {
    config: Config,
    capture: Capture,
    store: Arc<Mutex<Store>>,
    audio: AudioStore,
    state: Arc<Mutex<SessionState>>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct SessionState {
    readiness: Readiness,
    meeting_id: Option<i64>,
}

/// What the pipelines have learned, for the readiness panel.
#[derive(Debug, Default, Clone, Copy)]
struct PipelineStats {
    /// Decode seconds per second of audio.
    rtf: Option<f32>,
    /// Utterances discarded as speaker echo. Shown rather than hidden:
    /// suppression deletes speech, so a misfire has to be visible.
    echo_suppressed: u64,
}

type SharedStats = Arc<Mutex<PipelineStats>>;

impl Session {
    pub fn new(config: Config) -> Result<Self, AppError> {
        config.ensure_dirs()?;
        let store = Store::open(config.database())?;
        let readiness = Readiness::idle(config.missing_models());
        let config_audio_dir = config.audio_dir();

        let session = Self {
            config,
            capture: Capture::new(),
            store: Arc::new(Mutex::new(store)),
            audio: AudioStore::new(config_audio_dir),
            state: Arc::new(Mutex::new(SessionState {
                readiness,
                meeting_id: None,
            })),
            stop: Arc::new(AtomicBool::new(false)),
            workers: Vec::new(),
        };

        // A crash leaves an uncompressed recording behind, which would fill the
        // disk at several times the intended rate if nobody ever finished it.
        session.audio.compress_orphans();
        session.prune_audio();

        Ok(session)
    }

    /// Delete audio older than the retention the user chose.
    ///
    /// Transcripts are untouched: they are small and are the point, while audio
    /// is a means to re-processing and is what actually fills a disk.
    pub fn prune_audio(&self) -> u64 {
        let days = self.config.settings.audio_retention_days;
        let cutoff = jiff::Zoned::now()
            .checked_sub(jiff::Span::new().days(i64::from(days)))
            .ok()
            .map(|z| z.to_string());

        let Some(cutoff) = cutoff else {
            tracing::warn!(days, "cannot compute a retention cutoff");
            return 0;
        };

        let stale = match lock(&self.store).meetings_started_before(&cutoff) {
            Ok(ids) => ids,
            Err(err) => {
                tracing::error!("cannot find meetings to prune: {err}");
                return 0;
            }
        };
        self.audio.prune(&stale)
    }

    pub fn audio_usage_bytes(&self) -> u64 {
        self.audio.usage_bytes()
    }

    pub fn readiness(&self) -> Readiness {
        let mut readiness = lock(&self.state).readiness.clone();
        // Recomputed rather than served from the snapshot: a download that
        // finished since the last capture would otherwise leave the UI
        // believing a model is still missing, or worse, already there.
        if !self.is_running() {
            readiness.missing_models = self.config.missing_models();
        }
        readiness
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
        for stream in [&streams.mic, &streams.system] {
            if let Err(status) = stream {
                tracing::error!(
                    stream = %status.kind,
                    "capture source refused to start: {}",
                    status.error.as_deref().unwrap_or("no reason given")
                );
            }
        }

        if !streams.any_running() {
            // Silence here would leave a headless run with nothing to explain
            // why a meeting never happened.
            tracing::error!("no capture stream started; the meeting cannot begin");
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
        let pipeline_stats: SharedStats = Arc::new(Mutex::new(PipelineStats::default()));
        // One reference, shared: the system stream fills it and the microphone
        // checks itself against it.
        let echo = Arc::new(EchoReference::default());

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
                        Arc::clone(&pipeline_stats),
                        Arc::clone(&echo),
                    )?);
                }
                Err(status) => meters.push(Err(status)),
            }
        }

        self.workers
            .push(self.spawn_meter(app, meters, meeting_id, pipeline_stats));
        lock(&self.state).meeting_id = meeting_id;

        Ok(self.readiness())
    }

    /// Drain one capture ring into its pipeline, forwarding and storing results.
    fn spawn_stream_worker(
        &self,
        app: AppHandle,
        mut handle: StreamHandle,
        meeting_id: Option<i64>,
        stats: SharedStats,
        echo: Arc<EchoReference>,
    ) -> Result<JoinHandle<()>, AppError> {
        let kind = handle.info.kind;
        let stop = Arc::clone(&self.stop);
        let store = Arc::clone(&self.store);
        let config = self.config.clone();
        let audio = AudioStore::new(config.audio_dir());

        // Only build the heavy machinery when there is a meeting to transcribe;
        // a readiness check should not spend 600 MB and a model load.
        let pipeline = match meeting_id {
            Some(_) => {
                let transcriber: Arc<dyn Transcriber> = Arc::new(SherpaTranscriber::load(
                    &config.recognizer(),
                    config.asr_threads,
                )?);
                let mut pipeline = StreamPipeline::new(
                    kind,
                    handle.info.sample_rate,
                    &config.silero().to_string_lossy(),
                    transcriber,
                    VadSettings::default(),
                )?;

                // The system stream is the reference; the microphone is what
                // gets contaminated by it.
                // Keep the audio only if the user asked for it: with a
                // retention of zero there is nothing to re-process later, so
                // writing it would be pure cost.
                if let Some(meeting_id) = meeting_id
                    && config.settings.audio_retention_days > 0
                {
                    match audio.recorder(meeting_id, kind) {
                        Ok(recorder) => pipeline.record_to(Box::new(RecorderTap(recorder))),
                        Err(err) => {
                            tracing::error!(stream = %kind, "cannot record audio: {err}")
                        }
                    }
                }

                match kind {
                    StreamKind::System => {
                        pipeline.publish_echo_reference(echo);
                        // Only this stream carries several people, so only it
                        // needs a tracker. That asymmetry is the payoff for
                        // capturing the two sources separately.
                        match build_tracker(&config, &store) {
                            Ok(tracker) => pipeline.identify_speakers(tracker),
                            Err(err) => tracing::error!("speaker tracking disabled: {err}"),
                        }
                    }
                    StreamKind::Mic => pipeline.suppress_echo_of(echo),
                }
                Some(pipeline)
            }
            None => None,
        };

        Ok(thread::spawn(move || {
            let mut pipeline = pipeline;
            let mut buffer = vec![0.0f32; READ_CHUNK];

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
                if let Ok(mut slot) = stats.lock() {
                    slot.rtf = Some(pipeline.rtf());
                    // Only the microphone suppresses, so this never races.
                    if kind == StreamKind::Mic {
                        slot.echo_suppressed = pipeline.suppressed_echo();
                    }
                }
                // Re-render on every utterance rather than on a timer. A
                // rendered meeting is a few tens of kilobytes, and the
                // alternative is a file that trails the transcript by however
                // long the timer happens to be when the process dies.
                if emit_events(&app, &store, meeting_id, events) {
                    if let Some(id) = meeting_id {
                        persist_slots(&store, &config, id, pipeline.speaker_slots());
                        render_markdown(&store, &config, id);
                    }
                }
            }

            // Closing time: whatever speech is still buffered is still speech.
            if let Some(pipeline) = pipeline.as_mut() {
                let events = pipeline.flush();
                if emit_events(&app, &store, meeting_id, events) {
                    if let Some(id) = meeting_id {
                        persist_slots(&store, &config, id, pipeline.speaker_slots());
                        render_markdown(&store, &config, id);
                    }
                }
            }
        }))
    }

    /// Publish capture health at a steady rate, independent of audio activity.
    fn spawn_meter(
        &self,
        app: AppHandle,
        meters: Vec<Result<(shi_audio::SourceInfo, Arc<shi_audio::StreamStats>), StreamStatus>>,
        meeting_id: Option<i64>,
        stats: SharedStats,
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

                let sampled = stats.lock().map(|s| *s).unwrap_or_default();
                let readiness = Readiness {
                    mic,
                    system,
                    missing_models: missing.clone(),
                    recording: meeting_id.is_some(),
                    meeting_id,
                    rtf: sampled.rtf,
                    echo_suppressed: sampled.echo_suppressed,
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
                        echo_suppressed = readiness.echo_suppressed,
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

        let mut readiness = Readiness::idle(self.config.missing_models());
        readiness.meeting_id = meeting_id;
        self.set_readiness(readiness.clone());
        readiness
    }

    pub fn store(&self) -> Arc<Mutex<Store>> {
        Arc::clone(&self.store)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Apply and persist a settings change.
    ///
    /// Refused mid-meeting: swapping the recogniser under a running pipeline
    /// would leave one half of a transcript produced by a different model.
    pub fn apply_settings(&mut self, settings: crate::settings::Settings) -> Result<(), AppError> {
        if self.is_running() {
            return Err(AppError::BusyRecording);
        }
        let shortened =
            settings.audio_retention_days < self.config.settings.audio_retention_days;

        settings.save(&self.config.data_dir)?;
        self.config.settings = settings;
        self.set_readiness(Readiness::idle(self.config.missing_models()));

        // Shortening retention should free the disk now: a setting that only
        // takes effect on the next launch looks broken.
        if shortened {
            self.prune_audio();
        }
        Ok(())
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
///
/// Returns whether anything was written, so the caller knows the Markdown is
/// now stale.
fn emit_events(
    app: &AppHandle,
    store: &Arc<Mutex<Store>>,
    meeting_id: Option<i64>,
    events: Vec<PipelineEvent>,
) -> bool {
    let mut stored = false;

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
                speaker,
                ..
            } => {
                // Attribution is stored as a reference, never a copied name:
                // renaming someone later has to fix every line at once.
                let (speaker_id, slot, name, discovered) = match &speaker {
                    Some(Attribution::Known { speaker_id, name }) => {
                        (Some(*speaker_id), None, Some(name.clone()), None)
                    }
                    Some(Attribution::Slot { id, is_new }) => {
                        (None, Some(*id), None, is_new.then_some(*id))
                    }
                    Some(Attribution::Continuation { id }) => (None, Some(*id), None, None),
                    Some(Attribution::Unknown) | None => (None, None, None, None),
                };

                let segment = NewSegment {
                    stream,
                    start_ms: start.as_millis() as i64,
                    end_ms: end.as_millis() as i64,
                    speaker_id,
                    session_slot: slot,
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
                stored = true;

                if let Some(slot) = discovered
                    && app
                        .emit(TRANSCRIPT_EVENT, &TranscriptEvent::SpeakerDiscovered { slot })
                        .is_err()
                {
                    return stored;
                }

                TranscriptEvent::Final {
                    id,
                    stream: stream.as_str().into(),
                    start_ms: segment.start_ms,
                    end_ms: segment.end_ms,
                    speaker: name,
                    slot,
                    text,
                }
            }
        };

        if app.emit(TRANSCRIPT_EVENT, &payload).is_err() {
            return stored;
        }
    }

    stored
}

/// Build a tracker preloaded with every voice the user has already named.
fn build_tracker(config: &Config, store: &Arc<Mutex<Store>>) -> Result<SpeakerTracker, AppError> {
    let model_id = config.speaker_model_id();
    let mut tracker = SpeakerTracker::new(
        &config.speaker_model().to_string_lossy(),
        config.asr_threads,
        Thresholds::default(),
    )?;

    let voices = lock(store).voices_for_model(&model_id)?;
    let count = voices.len();
    tracker.load_profiles(
        voices
            .into_iter()
            .map(|voice| VoiceProfile {
                speaker_id: voice.speaker_id,
                name: voice.display_name,
                embeddings: voice.embeddings,
            })
            .collect(),
    );
    tracing::info!(known_voices = count, "speaker tracking ready");

    Ok(tracker)
}

/// Keep the stored voices in step with what the tracker has heard, so naming
/// one after the meeting has a centroid to turn into a voiceprint.
fn persist_slots(
    store: &Arc<Mutex<Store>>,
    config: &Config,
    meeting_id: i64,
    slots: &[shi_pipeline::SessionSlot],
) {
    let model_id = config.speaker_model_id();
    let store = lock(store);
    for slot in slots {
        let row = SessionSlot {
            meeting_id,
            slot: slot.id,
            centroid: slot.centroid().to_vec(),
            model_id: model_id.clone(),
            sample_path: None,
            total_speech_ms: slot.total_speech.as_millis() as i64,
            utterances: slot.utterances,
            resolved_speaker_id: None,
        };
        if let Err(err) = store.upsert_session_slot(&row) {
            tracing::error!("cannot store speaker slot {}: {err}", slot.id);
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
