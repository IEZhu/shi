use shi_audio::StreamKind;

/// A recorded meeting.
#[derive(Debug, Clone)]
pub struct Meeting {
    pub id: i64,
    pub title: String,
    /// RFC 3339 with offset, so the Markdown timestamps match the calendar.
    pub started_at: String,
    pub ended_at: Option<String>,
    pub stt_model_id: String,
    pub md_path: Option<String>,
    pub audio_dir: Option<String>,
}

/// One finalised utterance.
///
/// `speaker` is `None` until identification resolves it. Storing the name here
/// rather than in the Markdown is what makes retroactive renaming a single
/// UPDATE followed by a re-render.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub id: i64,
    pub stream: StreamKind,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: Option<String>,
    pub text: String,
}

/// A segment that has not been written yet.
#[derive(Debug, Clone, PartialEq)]
pub struct NewSegment {
    pub stream: StreamKind,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: Option<String>,
    pub text: String,
}
