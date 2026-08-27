//! Tauri shell for the meeting transcriber.
//!
//! M0 scope: prove both capture streams are alive before a meeting starts.
//! The readiness panel is not a nicety — on macOS a refused system-audio grant
//! looks exactly like a working stream that happens to be quiet, so the only
//! honest check is whether a non-zero sample has actually arrived.

mod capture;

use std::sync::Mutex;

use capture::{Capture, Readiness};
use tauri::{AppHandle, Manager, State};

struct AppState {
    capture: Mutex<Capture>,
}

/// Lock helper: a panicking metering thread must not wedge the whole UI, so we
/// recover the guard rather than propagating the poison.
fn capture_lock<'a>(state: &'a State<'_, AppState>) -> std::sync::MutexGuard<'a, Capture> {
    state
        .capture
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[tauri::command]
fn start_capture(app: AppHandle, state: State<'_, AppState>) -> Readiness {
    capture_lock(&state).start(app)
}

#[tauri::command]
fn stop_capture(state: State<'_, AppState>) -> Readiness {
    let mut capture = capture_lock(&state);
    capture.stop();
    capture.snapshot()
}

#[tauri::command]
fn readiness(state: State<'_, AppState>) -> Readiness {
    capture_lock(&state).snapshot()
}

/// Log to stderr by default, or to `SHI_LOG_FILE` when set — a bundled app
/// launched from Finder has nowhere else to put diagnostics.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "shi_app_lib=debug,shi_audio=debug".into());
    let builder = tracing_subscriber::fmt().with_env_filter(filter);

    match std::env::var("SHI_LOG_FILE").ok().and_then(|path| {
        std::fs::File::create(path).ok()
    }) {
        Some(file) => builder.with_writer(Mutex::new(file)).with_ansi(false).init(),
        None => builder.init(),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_tracing();

    tauri::Builder::default()
        .setup(|app| {
            app.manage(AppState {
                capture: Mutex::new(Capture::new()),
            });

            // A bundled app has no terminal to drive, so `SHI_AUTOSTART=1`
            // lets a test harness exercise capture end to end.
            if std::env::var("SHI_AUTOSTART").is_ok_and(|v| v == "1") {
                let handle = app.handle().clone();
                let state: State<'_, AppState> = handle.state();
                capture_lock(&state).start(handle.clone());
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_capture,
            stop_capture,
            readiness
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
