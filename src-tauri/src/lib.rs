//! Tauri shell for the meeting transcriber.
//!
//! Two capture streams feed two identical pipelines. Their finalised
//! utterances land in one SQLite database, which is then rendered to Markdown —
//! never appended to, so naming a voice afterwards corrects the whole file.

mod capture;
mod config;
mod error;
mod session;

use std::sync::Mutex;

use capture::Readiness;
use config::Config;
use error::AppError;
use serde::Serialize;
use session::Session;
use tauri::{AppHandle, Manager, RunEvent, State};

struct AppState {
    session: Mutex<Session>,
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

/// Give every segment currently labelled `from` the name `to`, then re-render.
///
/// Passing `null` for `from` claims the segments identification has not
/// resolved yet — the common case after a meeting.
#[tauri::command]
fn rename_speaker(
    state: State<'_, AppState>,
    meeting_id: i64,
    from: Option<String>,
    to: String,
) -> Result<usize, AppError> {
    let session = lock_session(&state);
    let store = session.store();
    let guard = store.lock().unwrap_or_else(|p| p.into_inner());

    let changed = guard.rename_speaker(meeting_id, from.as_deref(), &to)?;

    let meeting = guard.meeting(meeting_id)?;
    let segments = guard.segments(meeting_id)?;
    let config = session.config();
    let path = config.markdown_path(&meeting);
    shi_store::markdown::write_to(&path, &meeting, &segments, &config.markdown())?;

    Ok(changed)
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
                let mut session = lock_session(&state);
                if session.is_running() {
                    tracing::info!("closing the meeting before exit");
                    session.stop();
                }
            }
        });
}
