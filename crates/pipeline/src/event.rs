use std::time::Duration;

use shi_audio::StreamKind;

/// Something the pipeline learned about a stream, in the order it learned it.
///
/// Drafts are provisional and always superseded: every `Draft` for an
/// utterance is replaced by exactly one `Final` covering the same audio. The UI
/// shows drafts greyed out and swaps them for the final text in place.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    /// Re-decode of an utterance still being spoken. Cheap, imprecise, replaced.
    Draft {
        stream: StreamKind,
        start: Duration,
        text: String,
    },
    /// A completed utterance, closed by a pause. This is what gets persisted.
    Final {
        stream: StreamKind,
        start: Duration,
        end: Duration,
        text: String,
        /// Offsets of each token from the meeting start, when the model
        /// supplies them.
        token_offsets: Vec<Duration>,
        tokens: Vec<String>,
    },
    /// An utterance ended without producing any text, so the UI can drop the
    /// draft it is currently showing instead of leaving it stranded.
    DraftAbandoned { stream: StreamKind },
}

impl PipelineEvent {
    pub fn stream(&self) -> StreamKind {
        match self {
            PipelineEvent::Draft { stream, .. }
            | PipelineEvent::Final { stream, .. }
            | PipelineEvent::DraftAbandoned { stream } => *stream,
        }
    }
}
