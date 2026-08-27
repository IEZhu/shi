use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use shi_models::{CATALOGUE, ModelError, ModelSpec};
use tauri::{AppHandle, Emitter};

/// Event name carrying [`InstallProgress`] to the frontend.
pub const MODEL_EVENT: &str = "model";

/// One catalogue entry as the model manager shows it.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogueEntry {
    pub id: String,
    pub kind: shi_models::ModelKind,
    pub display_name: String,
    pub summary: String,
    pub languages: String,
    pub download_bytes: u64,
    pub installed: bool,
    /// True while this model is being fetched.
    pub installing: bool,
    /// Whether the recogniser choice currently points at this entry.
    pub selected: bool,
    /// Whether the checksum comes from the release itself or was recorded when
    /// the entry was added. Surfaced rather than hidden: the two say different
    /// things about what verification proves.
    pub checksum_published: bool,
}

/// How an install is going.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum InstallProgress {
    Downloading {
        id: String,
        downloaded: u64,
        total: u64,
    },
    Installed {
        id: String,
    },
    Cancelled {
        id: String,
    },
    Failed {
        id: String,
        message: String,
    },
}

/// Downloads currently in flight, so they can be reported and cancelled.
#[derive(Default)]
pub struct Downloads {
    active: Mutex<Vec<(String, Arc<AtomicBool>)>>,
}

impl Downloads {
    pub fn is_active(&self, id: &str) -> bool {
        self.lock().iter().any(|(active, _)| active == id)
    }

    fn begin(&self, id: &str) -> Option<Arc<AtomicBool>> {
        let mut active = self.lock();
        if active.iter().any(|(running, _)| running == id) {
            return None;
        }
        let flag = Arc::new(AtomicBool::new(true));
        active.push((id.to_string(), Arc::clone(&flag)));
        Some(flag)
    }

    fn finish(&self, id: &str) {
        self.lock().retain(|(running, _)| running != id);
    }

    /// Ask a running download to stop. Its partial file is kept, so the next
    /// attempt resumes rather than starting a 500 MB fetch over.
    pub fn cancel(&self, id: &str) {
        for (running, flag) in self.lock().iter() {
            if running == id {
                flag.store(false, Ordering::SeqCst);
            }
        }
    }

    pub fn cancel_all(&self) {
        for (_, flag) in self.lock().iter() {
            flag.store(false, Ordering::SeqCst);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(String, Arc<AtomicBool>)>> {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// What the model manager should display right now.
pub fn catalogue(
    models_dir: &PathBuf,
    downloads: &Downloads,
    selected_recognizer: &str,
) -> Vec<CatalogueEntry> {
    CATALOGUE
        .iter()
        .map(|spec| CatalogueEntry {
            id: spec.id.to_string(),
            kind: spec.kind,
            display_name: spec.display_name.to_string(),
            summary: spec.summary.to_string(),
            languages: spec.languages.to_string(),
            download_bytes: spec.download_bytes,
            installed: spec.installed(models_dir),
            installing: downloads.is_active(spec.id),
            selected: spec.id == selected_recognizer,
            checksum_published: spec.integrity.is_published(),
        })
        .collect()
}

/// Fetch a model on a background thread, reporting progress as it goes.
///
/// Returns immediately: a 500 MB download must not block the UI thread, and
/// the user should be able to keep using the readiness panel while it runs.
pub fn install_in_background(
    app: AppHandle,
    spec: &'static ModelSpec,
    models_dir: PathBuf,
    downloads: Arc<Downloads>,
) {
    let Some(keep_going) = downloads.begin(spec.id) else {
        // Already running; asking twice is not an error worth surfacing.
        return;
    };

    std::thread::spawn(move || {
        // Progress arrives per 256 KB chunk, which is far more often than a
        // human can read. Throttle to whole percents.
        let mut last_percent = u64::MAX;
        let emitter = app.clone();
        let id = spec.id.to_string();

        let result = shi_models::install(
            spec,
            &models_dir,
            |progress| {
                let percent = if progress.total == 0 {
                    0
                } else {
                    progress.downloaded * 100 / progress.total
                };
                if percent == last_percent {
                    return;
                }
                last_percent = percent;
                let _ = emitter.emit(
                    MODEL_EVENT,
                    &InstallProgress::Downloading {
                        id: id.clone(),
                        downloaded: progress.downloaded,
                        total: progress.total,
                    },
                );
            },
            || keep_going.load(Ordering::SeqCst),
        );

        downloads.finish(spec.id);

        let outcome = match result {
            Ok(_) => {
                tracing::info!(model = spec.id, "model ready");
                InstallProgress::Installed {
                    id: spec.id.to_string(),
                }
            }
            Err(ModelError::Cancelled) => InstallProgress::Cancelled {
                id: spec.id.to_string(),
            },
            Err(err) => {
                tracing::error!(model = spec.id, "install failed: {err}");
                InstallProgress::Failed {
                    id: spec.id.to_string(),
                    message: err.to_string(),
                }
            }
        };
        let _ = app.emit(MODEL_EVENT, &outcome);
    });
}
