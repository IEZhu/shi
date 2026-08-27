use shi_audio::StreamKind;

/// A recorded meeting.
#[derive(Debug, Clone)]
pub struct Meeting {
    pub id: i64,
    pub title: String,
    /// Zone-annotated timestamp, so wall-clock rendering survives a daylight
    /// saving change.
    pub started_at: String,
    pub ended_at: Option<String>,
    pub stt_model_id: String,
    pub md_path: Option<String>,
    pub audio_dir: Option<String>,
}

/// Someone the user has named.
#[derive(Debug, Clone, PartialEq)]
pub struct Speaker {
    pub id: i64,
    pub display_name: String,
    pub created_at: String,
    pub notes: Option<String>,
}

/// One finalised utterance.
///
/// Attribution is a reference, never a copied name: `speaker_id` for a known
/// person, `session_slot` for a voice this meeting has heard but cannot name
/// yet, and neither when it is too short to judge. Resolving the name at render
/// time is what makes renaming someone a single row rather than a rewrite of
/// every line they ever spoke.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub id: i64,
    pub stream: StreamKind,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker_id: Option<i64>,
    /// Resolved from `speakers` when the segment is read.
    pub speaker_name: Option<String>,
    pub session_slot: Option<u32>,
    pub text: String,
}

/// A segment that has not been written yet.
#[derive(Debug, Clone, PartialEq)]
pub struct NewSegment {
    pub stream: StreamKind,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker_id: Option<i64>,
    pub session_slot: Option<u32>,
    pub text: String,
}

/// An unnamed voice within one meeting.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSlot {
    pub meeting_id: i64,
    pub slot: u32,
    pub centroid: Vec<f32>,
    pub model_id: String,
    /// A few seconds of this voice, for the review screen to play back.
    pub sample_path: Option<String>,
    pub total_speech_ms: i64,
    pub utterances: u32,
    pub resolved_speaker_id: Option<i64>,
}

/// One stored example of a voice.
#[derive(Debug, Clone, PartialEq)]
pub struct Voiceprint {
    pub speaker_id: i64,
    pub embedding: Vec<f32>,
    pub model_id: String,
}

/// Everything known about one person's voice, ready for matching.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerVoice {
    pub speaker_id: i64,
    pub display_name: String,
    pub embeddings: Vec<Vec<f32>>,
}

/// Embeddings are stored as little-endian f32, which is what both the model
/// and every platform we target already use.
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
