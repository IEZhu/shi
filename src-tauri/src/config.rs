use std::path::{Path, PathBuf};

use shi_models::{Family, ModelKind, ModelSpec};
use shi_pipeline::ModelPaths;
use shi_store::{MarkdownOptions, Meeting};

use crate::settings::Settings;

/// Where the app keeps its data, and where it looks for models.
///
/// Every location can be overridden by an environment variable so a developer
/// can point at a checkout's `models/` directory without installing anything.
#[derive(Debug, Clone)]
pub struct Config {
    /// Database and other app-private state.
    pub data_dir: PathBuf,
    /// Downloaded recognisers and the VAD.
    pub models_dir: PathBuf,
    /// Where meeting Markdown is written, chosen by the user.
    pub markdown_dir: PathBuf,
    /// Threads per ASR instance. Two pipelines run at once, so this is not
    /// the whole machine.
    pub asr_threads: i32,
    /// The user's choices, reloaded whenever they change.
    pub settings: Settings,
}

impl Config {
    pub fn resolve() -> Self {
        let home = std::env::var_os("HOME").map(PathBuf::from);

        let data_dir = env_path("SHI_DATA_DIR").unwrap_or_else(|| {
            home.as_ref()
                .map(|h| h.join("Library/Application Support/io.github.iezhu.shi"))
                .unwrap_or_else(|| PathBuf::from(".shi"))
        });

        let models_dir = env_path("SHI_MODELS_DIR").unwrap_or_else(|| data_dir.join("models"));

        let markdown_dir = env_path("SHI_MARKDOWN_DIR").unwrap_or_else(|| {
            home.as_ref()
                .map(|h| h.join("Documents/Shi Meetings"))
                .unwrap_or_else(|| PathBuf::from("meetings"))
        });

        let settings = Settings::load(&data_dir);
        Self {
            data_dir,
            models_dir,
            markdown_dir,
            asr_threads: default_asr_threads(),
            settings,
        }
    }

    /// The recogniser the user picked, or the default if that is unavailable.
    pub fn recognizer_spec(&self) -> &'static ModelSpec {
        shi_models::by_id(&self.settings.recognizer_id)
            .filter(|spec| spec.kind == ModelKind::Recognizer)
            .or_else(|| {
                shi_models::CATALOGUE
                    .iter()
                    .find(|spec| spec.kind == ModelKind::Recognizer && spec.default)
            })
            .expect("the catalogue always contains a default recogniser")
    }

    pub fn database(&self) -> PathBuf {
        self.data_dir.join("meetings.db")
    }

    /// Where meeting audio is kept, separate from the database so it can be
    /// pruned, moved or excluded from a backup on its own.
    pub fn audio_dir(&self) -> PathBuf {
        self.data_dir.join("audio")
    }

    pub fn silero(&self) -> PathBuf {
        self.model_path("silero-vad")
    }

    /// The recogniser directory, laid out the way sherpa-onnx ships it.
    pub fn recognizer_dir(&self) -> PathBuf {
        self.recognizer_spec().path_in(&self.models_dir)
    }

    /// Paths in the shape the chosen model's family needs.
    pub fn recognizer(&self) -> ModelPaths {
        let spec = self.recognizer_spec();
        let dir = spec.path_in(&self.models_dir);
        match spec.family {
            Family::Whisper => {
                ModelPaths::whisper_int8(dir, spec.file_prefix.unwrap_or_default())
            }
            _ => ModelPaths::parakeet_int8(dir),
        }
    }

    /// The speaker-embedding model. NeMo TitaNet small was measured to be the
    /// clear winner over the CAM++ variants — see docs/speaker-identification.md.
    pub fn speaker_model(&self) -> PathBuf {
        self.model_path("titanet-small")
    }

    fn model_path(&self, id: &str) -> PathBuf {
        shi_models::by_id(id)
            .map(|spec| spec.path_in(&self.models_dir))
            .unwrap_or_else(|| self.models_dir.join(id))
    }

    /// Identifies which model produced a stored embedding. Embeddings from
    /// different models are not comparable, so they are never mixed.
    pub fn speaker_model_id(&self) -> String {
        self.speaker_model()
            .file_stem()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "speaker-model".into())
    }

    /// What is missing before a meeting can be transcribed.
    ///
    /// Reported in the readiness panel alongside the capture streams, because
    /// discovering a missing model at the start of a call is the same failure
    /// as discovering a missing permission.
    pub fn missing_models(&self) -> Vec<String> {
        shi_models::required(Some(&self.settings.recognizer_id))
            .into_iter()
            .filter(|spec| !spec.installed(&self.models_dir))
            .map(|spec| spec.display_name.to_string())
            .collect()
    }

    /// How meeting Markdown is rendered.
    pub fn markdown(&self) -> MarkdownOptions {
        MarkdownOptions::default()
    }

    /// Where a meeting's Markdown lives: dated, titled, and safe on any
    /// filesystem the user might sync the folder to.
    pub fn markdown_path(&self, meeting: &Meeting) -> PathBuf {
        let date = meeting
            .started_at
            .split('T')
            .next()
            .unwrap_or(&meeting.started_at);
        let title = sanitise_filename(&meeting.title);
        self.markdown_dir.join(format!("{date} {title}.md"))
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.markdown_dir)?;
        Ok(())
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| absolutise(&p))
}

/// Resolve relative overrides against the working directory, so
/// `SHI_MODELS_DIR=models` works from a checkout.
fn absolutise(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

/// Leave most of the machine free: two pipelines decode concurrently, and
/// diarization and the UI still need room.
fn default_asr_threads() -> i32 {
    if let Some(explicit) = std::env::var("SHI_ASR_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        return explicit.max(1);
    }
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    ((cores / 3) as i32).clamp(1, 6)
}

/// Strip what a filesystem — or a syncing client — would object to.
fn sanitise_filename(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();

    let trimmed = cleaned.trim().trim_matches('.').trim();
    if trimmed.is_empty() {
        "Встреча".to_string()
    } else {
        // Leave room for the date prefix and extension within the 255-byte
        // limit that most filesystems impose.
        trimmed.chars().take(120).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::sanitise_filename;

    #[test]
    fn path_separators_and_colons_cannot_reach_the_filename() {
        assert_eq!(sanitise_filename("Ретро: Q3/Q4"), "Ретро- Q3-Q4");
    }

    #[test]
    fn an_empty_title_still_yields_a_file() {
        assert_eq!(sanitise_filename("   "), "Встреча");
        assert_eq!(sanitise_filename("..."), "Встреча");
    }

    #[test]
    fn a_very_long_title_is_truncated_by_characters_not_bytes() {
        // Cyrillic is two bytes per character; truncating by bytes would split
        // one in half and produce an invalid name.
        let long = "я".repeat(400);
        let result = sanitise_filename(&long);
        assert_eq!(result.chars().count(), 120);
        assert!(result.is_char_boundary(result.len()));
    }
}
