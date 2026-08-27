use std::time::Duration;

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use crate::asr::ASR_SAMPLE_RATE;
use crate::error::{PipelineError, Result};

/// Similarity above which an utterance belongs to a voice we already know.
///
/// Measured, not guessed — see docs/speaker-identification.md. The two errors
/// are not symmetric: attaching the wrong name writes a confident lie into a
/// document the user will trust, while failing to recognise someone merely asks
/// them to name a speaker again. Both thresholds therefore sit high.
pub const DEFAULT_KNOWN_THRESHOLD: f32 = 0.75;

/// Similarity above which two utterances in one meeting are the same person.
/// Lower than `known`: splitting one participant into two slots costs a click,
/// while merging two people loses information.
pub const DEFAULT_SESSION_THRESHOLD: f32 = 0.70;

/// Utterances shorter than this are attributed but never define a voice: a
/// half-second of "угу" is not evidence of who is speaking.
const MIN_FOR_EVIDENCE: Duration = Duration::from_millis(700);

/// A new slot needs more than a match does. Creating one is the decision that
/// clutters the review screen, so it demands a real utterance behind it.
const MIN_FOR_NEW_SLOT: Duration = Duration::from_millis(1_000);

#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    pub known: f32,
    pub session: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            known: DEFAULT_KNOWN_THRESHOLD,
            session: DEFAULT_SESSION_THRESHOLD,
        }
    }
}

/// A voice the user has named, in some earlier meeting or this one.
///
/// Several embeddings per person on purpose: the same voice on AirPods and in a
/// conference room lands in different places, and matching takes the best of
/// them rather than an average that resembles neither.
#[derive(Debug, Clone)]
pub struct VoiceProfile {
    pub speaker_id: i64,
    pub name: String,
    pub embeddings: Vec<Vec<f32>>,
}

impl VoiceProfile {
    fn similarity(&self, embedding: &[f32]) -> f32 {
        self.embeddings
            .iter()
            .map(|known| cosine(known, embedding))
            .fold(f32::MIN, f32::max)
    }
}

/// An unnamed voice within the current meeting.
#[derive(Debug, Clone)]
pub struct SessionSlot {
    pub id: u32,
    /// Running mean of accepted embeddings, weighted by how much speech each
    /// contributed — recognition sharpens as the meeting goes on.
    centroid: Vec<f32>,
    weight: f32,
    pub total_speech: Duration,
    pub utterances: u32,
}

/// Who an utterance belongs to.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribution {
    /// Matched a named profile.
    Known { speaker_id: i64, name: String },
    /// Matched, or started, an unnamed voice in this meeting.
    Slot { id: u32, is_new: bool },
    /// Too short to judge; carried over from the previous utterance if there
    /// was one, because a two-word interjection almost always continues the
    /// current turn.
    Continuation { id: u32 },
    /// Too short to judge and nothing to continue.
    Unknown,
}

/// Tracks who is speaking on one stream.
pub struct SpeakerTracker {
    extractor: SpeakerEmbeddingExtractor,
    profiles: Vec<VoiceProfile>,
    slots: Vec<SessionSlot>,
    thresholds: Thresholds,
    next_slot: u32,
    last_slot: Option<u32>,
}

impl SpeakerTracker {
    pub fn new(model: &str, threads: i32, thresholds: Thresholds) -> Result<Self> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string()),
            num_threads: threads.max(1),
            debug: false,
            provider: None,
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .ok_or(PipelineError::SpeakerModelLoadFailed)?;

        Ok(Self {
            extractor,
            profiles: Vec::new(),
            slots: Vec::new(),
            thresholds,
            next_slot: 1,
            last_slot: None,
        })
    }

    /// Load the voices the user has already named.
    pub fn load_profiles(&mut self, profiles: Vec<VoiceProfile>) {
        self.profiles = profiles;
    }

    pub fn slots(&self) -> &[SessionSlot] {
        &self.slots
    }

    /// Embedding dimension, for storage.
    pub fn dim(&self) -> i32 {
        self.extractor.dim()
    }

    /// Compute the voice embedding of 16 kHz mono audio.
    pub fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(ASR_SAMPLE_RATE, samples);
        stream.input_finished();
        self.extractor.compute(&stream)
    }

    /// Decide who spoke an utterance, learning from it when it is long enough
    /// to be evidence.
    pub fn attribute(&mut self, samples: &[f32], duration: Duration) -> Attribution {
        if duration < MIN_FOR_EVIDENCE {
            return match self.last_slot {
                Some(id) => Attribution::Continuation { id },
                None => Attribution::Unknown,
            };
        }

        let Some(embedding) = self.embed(samples) else {
            return self
                .last_slot
                .map_or(Attribution::Unknown, |id| Attribution::Continuation { id });
        };

        // A named voice wins over an unnamed one: the user already told us who
        // this is, and re-asking would be worse than a rare mistake.
        if let Some(profile) = self
            .profiles
            .iter()
            .max_by(|a, b| a.similarity(&embedding).total_cmp(&b.similarity(&embedding)))
            .filter(|p| p.similarity(&embedding) >= self.thresholds.known)
        {
            return Attribution::Known {
                speaker_id: profile.speaker_id,
                name: profile.name.clone(),
            };
        }

        let best = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (index, cosine(&slot.centroid, &embedding)))
            .max_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((index, score)) = best
            && score >= self.thresholds.session
        {
            let slot = &mut self.slots[index];
            slot.absorb(&embedding, duration);
            self.last_slot = Some(slot.id);
            return Attribution::Slot {
                id: slot.id,
                is_new: false,
            };
        }

        if duration < MIN_FOR_NEW_SLOT {
            return self
                .last_slot
                .map_or(Attribution::Unknown, |id| Attribution::Continuation { id });
        }

        let id = self.next_slot;
        self.next_slot += 1;
        self.slots.push(SessionSlot {
            id,
            centroid: embedding,
            weight: duration.as_secs_f32(),
            total_speech: duration,
            utterances: 1,
        });
        self.last_slot = Some(id);

        Attribution::Slot { id, is_new: true }
    }

    /// Slots whose centroids are close enough that they are probably one
    /// person the tracker split in two, most likely because their voice
    /// changed partway through the meeting.
    ///
    /// Offered as suggestions on the review screen rather than merged
    /// silently: a wrong merge destroys information, a wrong suggestion costs
    /// a glance.
    pub fn merge_candidates(&self) -> Vec<(u32, u32, f32)> {
        let mut pairs = Vec::new();
        for (i, a) in self.slots.iter().enumerate() {
            for b in self.slots.iter().skip(i + 1) {
                let score = cosine(&a.centroid, &b.centroid);
                if score >= self.thresholds.session {
                    pairs.push((a.id, b.id, score));
                }
            }
        }
        pairs.sort_by(|x, y| y.2.total_cmp(&x.2));
        pairs
    }

    /// The centroid of a slot, for storing as a voice profile once named.
    pub fn centroid_of(&self, slot: u32) -> Option<&[f32]> {
        self.slots
            .iter()
            .find(|s| s.id == slot)
            .map(|s| s.centroid.as_slice())
    }
}

impl SessionSlot {
    /// The running mean embedding, for storing once this voice is named.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Fold an utterance into this voice, weighted by how much speech it holds.
    fn absorb(&mut self, embedding: &[f32], duration: Duration) {
        let weight = duration.as_secs_f32();
        let total = self.weight + weight;
        for (value, incoming) in self.centroid.iter_mut().zip(embedding) {
            *value = (*value * self.weight + incoming * weight) / total;
        }
        self.weight = total;
        self.total_speech += duration;
        self.utterances += 1;
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}
