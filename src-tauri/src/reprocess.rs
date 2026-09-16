use std::sync::{Arc, Mutex};

use shi_audio::StreamKind;
use shi_pipeline::{DEFAULT_SESSION_THRESHOLD, Span, SpeakerTracker, Thresholds, rediarize};
use shi_store::{AudioStore, SessionSlot, Store};

use crate::config::Config;
use crate::error::AppError;

/// What re-running speaker attribution changed.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Reprocessed {
    pub utterances: usize,
    pub speakers_before: usize,
    pub speakers_after: usize,
    pub renamed: usize,
}

/// Re-run speaker attribution over a finished meeting's recording.
///
/// The live tracker decides who is speaking with only the past to go on, so an
/// early utterance can open a voice that later evidence would have merged.
/// Working from the recording, every utterance is compared with every other at
/// once — strictly more information, and the reason keeping the audio is worth
/// the disk it costs.
pub fn speakers(
    store: &Arc<Mutex<Store>>,
    config: &Config,
    meeting_id: i64,
) -> Result<Reprocessed, AppError> {
    let audio = AudioStore::new(config.audio_dir())
        .read(meeting_id, StreamKind::System)
        .ok_or(AppError::NoRecording(meeting_id))?;

    let tracker = SpeakerTracker::new(
        &config.speaker_model().to_string_lossy(),
        config.asr_threads,
        Thresholds::default(),
    )?;

    let (spans, speakers_before) = {
        let guard = store.lock().unwrap_or_else(|p| p.into_inner());
        let segments = guard.segments(meeting_id)?;

        let spans: Vec<Span> = segments
            .iter()
            .filter(|segment| segment.stream == StreamKind::System)
            .map(|segment| Span {
                id: segment.id,
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
            })
            .collect();

        let before = segments
            .iter()
            .filter_map(|segment| segment.session_slot)
            .collect::<std::collections::HashSet<_>>()
            .len();

        (spans, before)
    };

    let assignments = rediarize(&tracker, &audio, &spans, None);
    if assignments.is_empty() {
        return Err(AppError::NothingToReprocess);
    }

    let speakers_after = assignments
        .iter()
        .map(|(_, slot)| *slot)
        .collect::<std::collections::HashSet<_>>()
        .len();

    // Re-embed for the centroids so each voice keeps a signature the review
    // screen can turn into a stored voiceprint.
    let embeddings: Vec<Vec<f32>> = assignments
        .iter()
        .filter_map(|(id, _)| {
            let span = spans.iter().find(|span| span.id == *id)?;
            let from = (span.start_ms.max(0) * 16) as usize;
            let to = ((span.end_ms.max(0) * 16) as usize).min(audio.len());
            (from < to).then(|| tracker.embed(&audio[from..to]))?
        })
        .collect();

    let model_id = config.speaker_model_id();
    let mut guard = store.lock().unwrap_or_else(|p| p.into_inner());
    let renamed = guard.reassign_slots(meeting_id, &assignments)?;

    if embeddings.len() == assignments.len() {
        for (slot, centroid) in shi_pipeline::centroids(&assignments, &embeddings) {
            guard.upsert_session_slot(&SessionSlot {
                meeting_id,
                slot,
                centroid,
                model_id: model_id.clone(),
                sample_path: None,
                total_speech_ms: 0,
                utterances: 0,
                resolved_speaker_id: None,
            })?;
        }
    }

    tracing::info!(
        meeting_id,
        utterances = assignments.len(),
        speakers_before,
        speakers_after,
        renamed,
        "re-ran speaker attribution from the recording"
    );

    Ok(Reprocessed {
        utterances: assignments.len(),
        speakers_before,
        speakers_after,
        renamed,
    })
}
