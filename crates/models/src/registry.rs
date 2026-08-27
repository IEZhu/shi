use std::path::{Path, PathBuf};

use serde::Serialize;

/// What a model is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    /// Turns speech into text.
    Recognizer,
    /// Finds where speech starts and stops.
    Vad,
    /// Turns a voice into a comparable vector.
    SpeakerEmbedding,
}

/// Which sherpa-onnx configuration a recogniser needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Family {
    NemoTransducer,
    Whisper,
    NotApplicable,
}

/// How a downloaded file is checked.
///
/// The distinction is deliberate and visible rather than flattened into one
/// "expected hash" field: only one of these actually attests to the upstream
/// artefact, and pretending otherwise would overstate what verification proves.
#[derive(Debug, Clone, Copy)]
pub enum Integrity {
    /// SHA-256 published by the release itself. Attests to the artefact.
    Published(&'static str),
    /// SHA-256 observed when this entry was added, for releases that publish
    /// no digest. Catches corruption and truncation; it says the file is the
    /// one we saw, not that anyone upstream vouches for it.
    Observed(&'static str),
}

impl Integrity {
    pub fn expected(&self) -> &'static str {
        match self {
            Integrity::Published(hash) | Integrity::Observed(hash) => hash,
        }
    }

    pub fn is_published(&self) -> bool {
        matches!(self, Integrity::Published(_))
    }
}

/// What arrives on disk.
#[derive(Debug, Clone, Copy)]
pub enum Install {
    /// A single file, saved under this name.
    File(&'static str),
    /// A `.tar.bz2` unpacked into the models directory, producing this folder.
    Archive(&'static str),
}

/// One downloadable model.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub id: &'static str,
    pub kind: ModelKind,
    pub family: Family,
    pub display_name: &'static str,
    /// One line the user can choose on.
    pub summary: &'static str,
    pub languages: &'static str,
    pub url: &'static str,
    /// Compressed size, which is what the progress bar counts.
    pub download_bytes: u64,
    pub integrity: Integrity,
    pub install: Install,
    /// Whisper packages prefix every file with the model size
    /// (`turbo-encoder.int8.onnx`), so the loader needs to know it.
    pub file_prefix: Option<&'static str>,
    /// Chosen when the user has expressed no preference.
    pub default: bool,
}

impl ModelSpec {
    /// Where this model lives once installed.
    pub fn path_in(&self, models_dir: &Path) -> PathBuf {
        match self.install {
            Install::File(name) | Install::Archive(name) => models_dir.join(name),
        }
    }

    /// Whether it is already there.
    pub fn installed(&self, models_dir: &Path) -> bool {
        let path = self.path_in(models_dir);
        match self.install {
            Install::File(_) => path.is_file(),
            // sherpa lays every recogniser out as a directory of ONNX files;
            // an empty directory left by an interrupted extraction is not an
            // installation.
            Install::Archive(_) => {
                path.is_dir()
                    && std::fs::read_dir(&path)
                        .map(|mut entries| entries.next().is_some())
                        .unwrap_or(false)
            }
        }
    }
}

/// Everything the app knows how to fetch.
///
/// Deliberately short. Every entry here has been run against real audio, and a
/// model nobody has listened to is not a choice worth offering.
pub const CATALOGUE: &[ModelSpec] = &[
    ModelSpec {
        id: "parakeet-tdt-0.6b-v3-int8",
        kind: ModelKind::Recognizer,
        family: Family::NemoTransducer,
        display_name: "Parakeet TDT 0.6B v3",
        summary: "Быстрее Whisper в разы, точнее на европейских языках",
        languages: "25 европейских языков, включая русский",
        url: concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
        ),
        download_bytes: 487_170_055,
        integrity: Integrity::Published(
            "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        ),
        install: Install::Archive("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"),
        file_prefix: None,
        default: true,
    },
    ModelSpec {
        id: "whisper-turbo",
        kind: ModelKind::Recognizer,
        family: Family::Whisper,
        display_name: "Whisper turbo",
        summary: "Языков больше всех, но заметно медленнее Parakeet",
        languages: "100+ языков",
        url: concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            "sherpa-onnx-whisper-turbo.tar.bz2"
        ),
        download_bytes: 563_790_207,
        integrity: Integrity::Observed(
            "b11acbbcd660b44a8e0df33724feb5aaa709cf65668f2823d59f656312544f22",
        ),
        install: Install::Archive("sherpa-onnx-whisper-turbo"),
        file_prefix: Some("turbo"),
        default: false,
    },
    ModelSpec {
        id: "whisper-tiny",
        kind: ModelKind::Recognizer,
        family: Family::Whisper,
        display_name: "Whisper tiny",
        summary: "Для слабых машин и быстрой проверки; точность заметно ниже",
        languages: "100+ языков",
        url: concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            "sherpa-onnx-whisper-tiny.tar.bz2"
        ),
        download_bytes: 116_204_861,
        integrity: Integrity::Observed(
            "c46116994e539aa165266d96b325252728429c12535eb9d8b6a2b10f129e66b1",
        ),
        install: Install::Archive("sherpa-onnx-whisper-tiny"),
        file_prefix: Some("tiny"),
        default: false,
    },
    ModelSpec {
        id: "silero-vad",
        kind: ModelKind::Vad,
        family: Family::NotApplicable,
        display_name: "Silero VAD",
        summary: "Определяет границы реплик",
        languages: "любой язык",
        url: concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/",
            "silero_vad.onnx"
        ),
        download_bytes: 643_854,
        integrity: Integrity::Published(
            "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        ),
        install: Install::File("silero_vad.onnx"),
        file_prefix: None,
        default: true,
    },
    ModelSpec {
        id: "titanet-small",
        kind: ModelKind::SpeakerEmbedding,
        family: Family::NotApplicable,
        display_name: "NeMo TitaNet small",
        summary: "Различает голоса; шире всех разделяет похожие",
        languages: "не зависит от языка",
        url: concat!(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/",
            "speaker-recongition-models/nemo_en_titanet_small.onnx"
        ),
        download_bytes: 40_257_283,
        integrity: Integrity::Observed(
            "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
        ),
        install: Install::File("nemo_en_titanet_small.onnx"),
        file_prefix: None,
        default: true,
    },
];

pub fn by_id(id: &str) -> Option<&'static ModelSpec> {
    CATALOGUE.iter().find(|spec| spec.id == id)
}

/// The models a meeting cannot start without.
pub fn required(preferred_recognizer: Option<&str>) -> Vec<&'static ModelSpec> {
    let recognizer = preferred_recognizer
        .and_then(by_id)
        .filter(|spec| spec.kind == ModelKind::Recognizer)
        .or_else(|| {
            CATALOGUE
                .iter()
                .find(|spec| spec.kind == ModelKind::Recognizer && spec.default)
        });

    CATALOGUE
        .iter()
        .filter(|spec| spec.kind != ModelKind::Recognizer && spec.default)
        .chain(recognizer)
        .collect()
}
