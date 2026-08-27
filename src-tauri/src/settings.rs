use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Choices the user has made that must outlive the process.
///
/// A small JSON file rather than a table: these are a handful of scalars read
/// once at startup, and putting them in the meetings database would tie a
/// preference change to a schema migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Catalogue id of the recogniser to use. An unknown id falls back to the
    /// default rather than refusing to start.
    pub recognizer_id: String,
    /// Whether to show draft text while someone is still speaking.
    pub show_drafts: bool,
    /// Discard microphone audio that is the speakers coming back in.
    pub suppress_echo: bool,
    /// Days to keep meeting audio. Zero means keep none, which turns off
    /// re-diarization and re-transcription but costs no disk.
    pub audio_retention_days: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            recognizer_id: "parakeet-tdt-0.6b-v3-int8".into(),
            show_drafts: true,
            suppress_echo: true,
            // Long enough to revisit last sprint's meetings, short enough that
            // the folder does not quietly become the largest thing on the disk.
            audio_retention_days: 14,
        }
    }
}

impl Settings {
    pub fn path_in(data_dir: &Path) -> PathBuf {
        data_dir.join("settings.json")
    }

    /// Read settings, falling back to defaults for anything unreadable.
    ///
    /// A corrupt settings file must not stop the app: losing a preference is a
    /// nuisance, refusing to launch is not.
    pub fn load(data_dir: &Path) -> Self {
        let path = Self::path_in(data_dir);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|err| {
                tracing::warn!("settings at {} are unreadable, using defaults: {err}", path.display());
                Self::default()
            }),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(err) => {
                tracing::warn!("cannot read settings: {err}");
                Self::default()
            }
        }
    }

    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(data_dir)?;
        let text = serde_json::to_string_pretty(self)?;
        // Write beside and rename, so an interrupted save cannot leave a
        // half-written file that the next launch refuses to parse.
        let path = Self::path_in(data_dir);
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, &path)
    }
}
