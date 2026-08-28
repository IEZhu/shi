use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;
use shi_audio::{StreamHandle, StreamKind};
use shi_pipeline::{
    Attribution, AudioTap, EchoReference, LiveNames, PipelineEvent, SherpaTranscriber, SpeakerTracker,
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
///
/// `rename_all_fields` is load-bearing: on an internally tagged enum,
/// `rename_all` renames the *variants* only. Without it `start_ms` reached the
/// UI unchanged while single-word fields happened to match, so timestamps —
/// and only timestamps — arrived as undefined.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
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
/// Whether something already running has to be torn down and started again.
///
/// The readiness check and a meeting both hold the capture devices, so
/// "already running" is not the same question as "already doing what was
/// asked". A check that is running when the user presses record must be
/// upgraded: treating it as started is what made the button do nothing at all.
fn wants_upgrade(recording: bool, wants_meeting: bool) -> bool {
    wants_meeting && !recording
}

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
    /// Voices the user names while the meeting is still running.
    live_names: LiveNames,
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
            live_names: Arc::new(Mutex::new(Vec::new())),
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
            let recording = lock(&self.state).meeting_id.is_some();
            if !wants_upgrade(recording, title.is_some()) {
                return Ok(self.readiness());
            }
            tracing::debug!("upgrading a readiness check into a meeting");
            self.stop();
        }

        // Load the recogniser before opening the microphone, not after. It
        // takes seconds, and any audio captured while it loads is audio the
        // user spoke after pressing record — it has to be either kept and
        // dated correctly, or not captured at all. Loading first makes it the
        // latter, and leaves nothing to throw away.
        //
        // One recogniser, shared. Loading it per stream cost 650 MB twice and,
        // worse, started the two workers seconds apart — which put their
        // transcripts on different timelines and left the echo detector
        // comparing the wrong moments. sherpa-onnx declares the recogniser
        // `Sync`, so sharing it is what the library intends.
        let transcriber: Option<Arc<dyn Transcriber>> = match &title {
            Some(_) => Some(Arc::new(SherpaTranscriber::load(
                &self.config.recognizer(),
                self.config.asr_threads,
            )?)),
            None => None,
        };

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

        let StartedStreams { mut mic, mut system } = streams;

        // The two sources still do not open at the same instant — the system
        // tap waits to see a first frame — so each ring holds a different
        // amount of audio by now. Kept, that backlog would stamp itself at time
        // zero and put the two streams on timelines 0.93 s apart, measured,
        // which is more than twice the window the echo detector may search.
        // Drop it, then start the clock.
        for handle in [&mut mic, &mut system] {
            if let Ok(handle) = handle {
                let dropped = handle.discard_backlog();
                tracing::debug!(
                    stream = %handle.info.kind,
                    samples = dropped,
                    "discarded pre-meeting backlog"
                );
            }
        }
        let origin = Instant::now();

        // A meeting starts with nobody named yet.
        if let Ok(mut pending) = self.live_names.lock() {
            pending.clear();
        }

        self.stop.store(false, Ordering::SeqCst);
        let pipeline_stats: SharedStats = Arc::new(Mutex::new(PipelineStats::default()));
        // One reference, shared: the system stream fills it and the microphone
        // checks itself against it.
        let echo = Arc::new(EchoReference::default());

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
                        origin,
                        transcriber.clone(),
                        Arc::clone(&self.live_names),
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
        origin: Instant,
        transcriber: Option<Arc<dyn Transcriber>>,
        live_names: LiveNames,
    ) -> Result<JoinHandle<()>, AppError> {
        let kind = handle.info.kind;
        let stop = Arc::clone(&self.stop);
        let store = Arc::clone(&self.store);
        let config = self.config.clone();
        let audio = AudioStore::new(config.audio_dir());

        // Only build the heavy machinery when there is a meeting to transcribe;
        // a readiness check should not spend 600 MB and a model load.
        let pipeline = match transcriber {
            Some(transcriber) => {
                let mut pipeline = StreamPipeline::new(
                    kind,
                    handle.info.sample_rate,
                    &config.silero().to_string_lossy(),
                    transcriber,
                    VadSettings::default(),
                )?;
                // Both streams share one origin, so their transcripts line up
                // and the echo detector is comparing the same moment.
                pipeline.started_at(origin);

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

                // The system stream is the reference; the microphone is what
                // gets contaminated by it.
                match kind {
                    StreamKind::System => {
                        pipeline.publish_echo_reference(echo);
                        // Only this stream carries several people, so only it
                        // needs a tracker. That asymmetry is the payoff for
                        // capturing the two sources separately.
                        match build_tracker(&config, &store) {
                            Ok(tracker) => {
                                pipeline.identify_speakers(tracker);
                                pipeline.accept_names_from(live_names);
                            }
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

    /// Tell a running meeting that a voice now has a name.
    ///
    /// Without this the database and the transcript agree about the past and
    /// disagree about everything said afterwards: the tracker keeps calling
    /// that person by their speaker number for the rest of the call.
    pub fn note_named_voice(&self, slot: u32, speaker_id: i64, name: &str) {
        if let Ok(mut pending) = self.live_names.lock() {
            pending.push((slot, speaker_id, name.to_string()));
        }
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

#[cfg(test)]
mod tests {
    #[test]
    fn a_running_check_is_upgraded_when_the_user_presses_record() {
        assert!(super::wants_upgrade(false, true));
    }

    #[test]
    fn a_running_meeting_is_left_alone() {
        assert!(!super::wants_upgrade(true, true));
        assert!(!super::wants_upgrade(true, false));
    }

    #[test]
    fn checking_twice_does_not_restart_capture() {
        assert!(!super::wants_upgrade(false, false));
    }

    use super::*;

    /// The frontend reads these names directly, and a mismatch is invisible
    /// from Rust: it shows up as `undefined` in the UI. On an internally
    /// tagged enum `rename_all` renames only the variants, so timestamps once
    /// arrived as `start_ms` and rendered as NaN while every single-word field
    /// happened to line up.
    #[test]
    fn transcript_events_use_the_names_the_frontend_expects() {
        let event = TranscriptEvent::Final {
            id: 1,
            stream: "system".into(),
            start_ms: 1_234,
            end_ms: 5_678,
            speaker: Some("Мария".into()),
            slot: Some(2),
            text: "Привет.".into(),
        };

        let json = serde_json::to_value(&event).expect("serialise");
        let object = json.as_object().expect("an object");

        for field in ["kind", "id", "stream", "startMs", "endMs", "speaker", "slot", "text"] {
            assert!(object.contains_key(field), "missing {field} in {json}");
        }
        assert!(!object.contains_key("start_ms"), "snake_case leaked: {json}");
        assert_eq!(object["kind"], "final");
        assert_eq!(object["startMs"], 1_234);
    }

    #[test]
    fn draft_events_use_the_same_names() {
        let json = serde_json::to_value(TranscriptEvent::Draft {
            stream: "mic".into(),
            start_ms: 42,
            text: "…".into(),
        })
        .expect("serialise");

        assert_eq!(json["kind"], "draft");
        assert_eq!(json["startMs"], 42);
        assert!(json.get("start_ms").is_none());
    }

    #[test]
    fn speaker_discovery_reaches_the_frontend_by_that_name() {
        let json =
            serde_json::to_value(TranscriptEvent::SpeakerDiscovered { slot: 3 }).expect("serialise");
        assert_eq!(json["kind"], "speakerDiscovered");
        assert_eq!(json["slot"], 3);
    }
}
