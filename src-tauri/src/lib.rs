//! Tauri shell for the meeting transcriber.
//!
//! Two capture streams feed two identical pipelines. Their finalised
//! utterances land in one SQLite database, which is then rendered to Markdown —
//! never appended to, so naming a voice afterwards corrects the whole file.

mod capture;
mod config;
mod error;
mod models;
mod reprocess;
mod session;
mod settings;

use std::sync::{Arc, Mutex};

use capture::Readiness;
use config::Config;
use error::AppError;
use serde::Serialize;
use models::{CatalogueEntry, Downloads};
use session::Session;
use settings::Settings;
use tauri::{AppHandle, Manager, RunEvent, State};

struct AppState {
    session: Mutex<Session>,
    downloads: Arc<Downloads>,
}

/// A worker panic must not wedge the UI, so recover the guard rather than
/// propagating the poison.
fn lock_session<'a>(state: &'a State<'_, AppState>) -> std::sync::MutexGuard<'a, Session> {
    state
        .session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A past meeting, for the archive list.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MeetingSummary {
    id: i64,
    title: String,
    started_at: String,
    ended_at: Option<String>,
    md_path: Option<String>,
    segment_count: usize,
}

#[tauri::command]
fn readiness(state: State<'_, AppState>) -> Readiness {
    lock_session(&state).readiness()
}

/// Start capture only — the pre-meeting check. Loads no models.
#[tauri::command]
fn start_check(app: AppHandle, state: State<'_, AppState>) -> Result<Readiness, AppError> {
    lock_session(&state).start_check(app)
}

#[tauri::command]
fn start_meeting(
    app: AppHandle,
    state: State<'_, AppState>,
    title: String,
) -> Result<Readiness, AppError> {
    let title = if title.trim().is_empty() {
        default_meeting_title()
    } else {
        title
    };
    lock_session(&state).start_meeting(app, &title)
}

#[tauri::command]
fn stop(state: State<'_, AppState>) -> Readiness {
    lock_session(&state).stop()
}

#[tauri::command]
fn meetings(state: State<'_, AppState>) -> Result<Vec<MeetingSummary>, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let store = store.lock().unwrap_or_else(|p| p.into_inner());

    store
        .meetings()?
        .into_iter()
        .map(|meeting| {
            let segment_count = store.segments(meeting.id).map(|s| s.len()).unwrap_or(0);
            Ok(MeetingSummary {
                id: meeting.id,
                title: meeting.title,
                started_at: meeting.started_at,
                ended_at: meeting.ended_at,
                md_path: meeting.md_path,
                segment_count,
            })
        })
        .collect()
}

/// An unnamed voice heard in a meeting, for the review screen.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UnnamedVoice {
    slot: u32,
    total_speech_ms: i64,
    utterances: u32,
    /// A line this voice actually said, so the user can recognise it without
    /// playing anything back.
    excerpt: Option<String>,
}

#[tauri::command]
fn unnamed_voices(
    state: State<'_, AppState>,
    meeting_id: i64,
) -> Result<Vec<UnnamedVoice>, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let store = store.lock().unwrap_or_else(|p| p.into_inner());

    let segments = store.segments(meeting_id)?;
    Ok(store
        .session_slots(meeting_id)?
        .into_iter()
        .filter(|slot| slot.resolved_speaker_id.is_none())
        .map(|slot| UnnamedVoice {
            slot: slot.slot,
            total_speech_ms: slot.total_speech_ms,
            utterances: slot.utterances,
            // The longest thing they said carries the most recognisable
            // content; a two-word line identifies nobody.
            excerpt: segments
                .iter()
                .filter(|s| s.session_slot == Some(slot.slot))
                .max_by_key(|s| s.text.chars().count())
                .map(|s| s.text.clone()),
        })
        .collect())
}

/// Give an unnamed voice a name.
///
/// Claims every line it spoke, and stores its voiceprint so the same person is
/// recognised automatically in later meetings.
#[tauri::command]
fn name_voice(
    state: State<'_, AppState>,
    meeting_id: i64,
    slot: u32,
    name: String,
) -> Result<i64, AppError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(AppError::EmptyName);
    }

    // Enrolment outlives the meeting, so it is worth an audit line.
    tracing::info!(meeting_id, slot, name, "naming a voice");

    let session = lock_session(&state);
    let store = session.store();
    let speaker = {
        let mut guard = store.lock().unwrap_or_else(|p| p.into_inner());
        guard.name_session_slot(meeting_id, slot, name, &jiff::Zoned::now().to_string())?
    };

    rerender(&session, meeting_id)?;
    Ok(speaker.id)
}

/// Undo an enrolment.
///
/// A voiceprint recorded under the wrong name does not merely spoil one
/// transcript — it will confidently mislabel every future meeting that person
/// attends. Removing it has to be as easy as creating it was.
#[tauri::command]
fn forget_speaker(state: State<'_, AppState>, speaker_id: i64) -> Result<(), AppError> {
    tracing::info!(speaker_id, "forgetting a voice");

    let session = lock_session(&state);
    let store = session.store();
    let meetings = {
        let guard = store.lock().unwrap_or_else(|p| p.into_inner());
        guard.forget_speaker(speaker_id)?;
        guard.meetings()?
    };

    for meeting in meetings {
        rerender(&session, meeting.id)?;
    }
    Ok(())
}

/// Someone the app has learned to recognise.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct KnownSpeaker {
    id: i64,
    display_name: String,
}

#[tauri::command]
fn speakers(state: State<'_, AppState>) -> Result<Vec<KnownSpeaker>, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());
    Ok(guard
        .speakers()?
        .into_iter()
        .map(|speaker| KnownSpeaker {
            id: speaker.id,
            display_name: speaker.display_name,
        })
        .collect())
}

/// Rename a person everywhere they appear.
///
/// One row: every transcript they were ever in is correct the next time it
/// renders, which is why segments store a reference rather than a name.
#[tauri::command]
fn rename_speaker(
    state: State<'_, AppState>,
    speaker_id: i64,
    name: String,
) -> Result<(), AppError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(AppError::EmptyName);
    }

    let session = lock_session(&state);
    let store = session.store();
    let meetings = {
        let guard = store.lock().unwrap_or_else(|p| p.into_inner());
        guard.rename_speaker(speaker_id, name)?;
        guard.meetings()?
    };

    // Every meeting they appear in now renders differently.
    for meeting in meetings {
        rerender(&session, meeting.id)?;
    }
    Ok(())
}

/// Rewrite a meeting's Markdown from the database.
fn rerender(session: &Session, meeting_id: i64) -> Result<(), AppError> {
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());
    let meeting = guard.meeting(meeting_id)?;
    let segments = guard.segments(meeting_id)?;
    let config = session.config();
    let path = config.markdown_path(&meeting);
    shi_store::markdown::write_to(&path, &meeting, &segments, &config.markdown())?;
    Ok(())
}

// ---- archive -----------------------------------------------------------

/// A line of a past meeting, for the archive view.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArchivedLine {
    id: i64,
    stream: String,
    start_ms: i64,
    speaker: Option<String>,
    slot: Option<u32>,
    text: String,
}

/// One search result.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult {
    meeting_id: i64,
    meeting_title: String,
    meeting_started_at: String,
    start_ms: i64,
    speaker: Option<String>,
    slot: Option<u32>,
    /// The matched line, with matches wrapped in the markers the UI splits on.
    snippet: String,
}

#[tauri::command]
fn meeting_transcript(
    state: State<'_, AppState>,
    meeting_id: i64,
) -> Result<Vec<ArchivedLine>, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());

    Ok(guard
        .segments(meeting_id)?
        .into_iter()
        .map(|segment| ArchivedLine {
            id: segment.id,
            stream: segment.stream.as_str().into(),
            start_ms: segment.start_ms,
            speaker: segment.speaker_name,
            slot: segment.session_slot,
            text: segment.text,
        })
        .collect())
}

/// Search every transcript.
#[tauri::command]
fn search(state: State<'_, AppState>, query: String) -> Result<Vec<SearchResult>, AppError> {
    const LIMIT: usize = 60;

    let session = lock_session(&state);
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());

    Ok(guard
        .search(&query, LIMIT)?
        .into_iter()
        .map(|hit| SearchResult {
            meeting_id: hit.meeting_id,
            meeting_title: hit.meeting_title,
            meeting_started_at: hit.meeting_started_at,
            start_ms: hit.start_ms,
            speaker: hit.speaker_name,
            slot: hit.session_slot,
            snippet: hit.snippet,
        })
        .collect())
}

/// Delete a meeting and its transcript. The Markdown file is left alone: it is
/// in the user's own folder and may already have been edited or filed.
#[tauri::command]
fn delete_meeting(state: State<'_, AppState>, meeting_id: i64) -> Result<(), AppError> {
    tracing::info!(meeting_id, "deleting a meeting");
    let session = lock_session(&state);
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());
    guard.delete_meeting(meeting_id)?;
    Ok(())
}

/// Re-run speaker attribution over a finished meeting's recording.
#[tauri::command]
fn reprocess_speakers(
    state: State<'_, AppState>,
    meeting_id: i64,
) -> Result<reprocess::Reprocessed, AppError> {
    let session = lock_session(&state);
    let outcome = reprocess::speakers(&session.store(), session.config(), meeting_id)?;
    rerender(&session, meeting_id)?;
    Ok(outcome)
}

// ---- storage -----------------------------------------------------------

/// What the app is using on disk, and for how long it keeps it.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StorageUsage {
    audio_bytes: u64,
    retention_days: u32,
    meetings: usize,
}

#[tauri::command]
fn storage_usage(state: State<'_, AppState>) -> Result<StorageUsage, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let meetings = {
        let guard = store.lock().unwrap_or_else(|p| p.into_inner());
        guard.meetings()?.len()
    };

    Ok(StorageUsage {
        audio_bytes: session.audio_usage_bytes(),
        retention_days: session.config().settings.audio_retention_days,
        meetings,
    })
}

/// Apply the retention setting now rather than waiting for the next launch.
#[tauri::command]
fn prune_audio(state: State<'_, AppState>) -> u64 {
    lock_session(&state).prune_audio()
}

// ---- models ------------------------------------------------------------

#[tauri::command]
fn model_catalogue(state: State<'_, AppState>) -> Vec<CatalogueEntry> {
    let session = lock_session(&state);
    let config = session.config();
    models::catalogue(
        &config.models_dir,
        &state.downloads,
        &config.settings.recognizer_id,
    )
}

/// Start fetching a model. Returns at once; progress arrives as events.
#[tauri::command]
fn install_model(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), AppError> {
    let spec = shi_models::by_id(&id).ok_or_else(|| AppError::UnknownModel(id.clone()))?;
    let models_dir = lock_session(&state).config().models_dir.clone();
    models::install_in_background(app, spec, models_dir, Arc::clone(&state.downloads));
    Ok(())
}

/// Stop a running download. The partial file stays, so retrying resumes.
#[tauri::command]
fn cancel_model_install(state: State<'_, AppState>, id: String) {
    state.downloads.cancel(&id);
}

#[tauri::command]
fn uninstall_model(state: State<'_, AppState>, id: String) -> Result<(), AppError> {
    let spec = shi_models::by_id(&id).ok_or_else(|| AppError::UnknownModel(id.clone()))?;
    let models_dir = lock_session(&state).config().models_dir.clone();
    shi_models::uninstall(spec, &models_dir)?;
    tracing::info!(model = %id, "model removed");
    Ok(())
}

// ---- settings ----------------------------------------------------------

#[tauri::command]
fn settings(state: State<'_, AppState>) -> Settings {
    lock_session(&state).config().settings.clone()
}

/// Apply and persist a settings change.
#[tauri::command]
fn save_settings(state: State<'_, AppState>, settings: Settings) -> Result<Settings, AppError> {
    let mut session = lock_session(&state);
    session.apply_settings(settings)?;
    Ok(session.config().settings.clone())
}

fn default_meeting_title() -> String {
    jiff::Zoned::now().strftime("Встреча %d.%m %H:%M").to_string()
}

/// Log to stderr by default, or to `SHI_LOG_FILE` when set — a bundled app
/// launched from Finder has nowhere else to put diagnostics.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        "shi_app_lib=debug,shi_audio=debug,shi_pipeline=debug,shi_store=info".into()
    });
    let builder = tracing_subscriber::fmt().with_env_filter(filter);

    match std::env::var("SHI_LOG_FILE")
        .ok()
        .and_then(|path| std::fs::File::create(path).ok())
    {
        Some(file) => builder.with_writer(Mutex::new(file)).with_ansi(false).init(),
        None => builder.init(),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_tracing();
    let config = Config::resolve();
    tracing::info!(
        data = %config.data_dir.display(),
        models = %config.models_dir.display(),
        markdown = %config.markdown_dir.display(),
        asr_threads = config.asr_threads,
        "starting"
    );

    tauri::Builder::default()
        .setup(move |app| {
            let session = Session::new(config)?;
            app.manage(AppState {
                session: Mutex::new(session),
                downloads: Arc::new(Downloads::default()),
            });

            // A bundled app has no terminal to drive, so these let a test
            // harness exercise the whole chain end to end.
            let handle = app.handle().clone();
            match std::env::var("SHI_AUTOSTART").as_deref() {
                Ok("check") => {
                    let state: State<'_, AppState> = handle.state();
                    let _ = lock_session(&state).start_check(handle.clone());
                }
                Ok("meeting") => {
                    let state: State<'_, AppState> = handle.state();
                    let title = std::env::var("SHI_MEETING_TITLE")
                        .unwrap_or_else(|_| default_meeting_title());
                    if let Err(err) = lock_session(&state).start_meeting(handle.clone(), &title) {
                        tracing::error!("autostart failed: {err}");
                    }
                }
                _ => {}
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            readiness,
            start_check,
            start_meeting,
            stop,
            meetings,
            meeting_transcript,
            search,
            delete_meeting,
            reprocess_speakers,
            storage_usage,
            prune_audio,
            model_catalogue,
            install_model,
            cancel_model_install,
            uninstall_model,
            settings,
            save_settings,
            unnamed_voices,
            name_voice,
            forget_speaker,
            speakers,
            rename_speaker
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|handle, event| {
            // Quitting mid-meeting must still close it: otherwise the meeting
            // is left open in the database and its Markdown stops at whichever
            // periodic render happened last. Dropping managed state on exit is
            // not guaranteed to run, so do it explicitly.
            // Which of these macOS delivers depends on how the quit was
            // requested, so handle both; `stop` is a no-op once stopped.
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                let state: State<'_, AppState> = handle.state();
                // Downloads keep their partial files, so quitting mid-fetch
                // costs nothing but the time already spent.
                state.downloads.cancel_all();

                let mut session = lock_session(&state);
                if session.is_running() {
                    tracing::info!("closing the meeting before exit");
                    session.stop();
                }
            }
        });
}
