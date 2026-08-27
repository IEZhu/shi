use std::sync::{Arc, Mutex};
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
    /// Set once this voice has been recognised as somebody already named.
    ///
    /// Identity has to hold for a whole meeting. Checking every utterance
    /// against the stored profile independently makes the same person appear
    /// as "Мария" on one line and "Спикер 1" on the next, because utterances
    /// vary and some land below the threshold. The first confident match binds
    /// the name to the voice; the rest follow the voice.
    pub known: Option<(i64, String)>,
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

/// Names given to voices while the meeting is still running.
///
/// Naming a voice mid-meeting writes to the database and fixes the lines
/// already spoken, but the tracker is running on another thread and would
/// otherwise keep calling that person "Спикер 2" for the rest of the call.
pub type LiveNames = Arc<Mutex<Vec<(u32, i64, String)>>>;

/// Tracks who is speaking on one stream.
pub struct SpeakerTracker {
    extractor: SpeakerEmbeddingExtractor,
    profiles: Vec<VoiceProfile>,
    slots: Vec<SessionSlot>,
    thresholds: Thresholds,
    next_slot: u32,
    last_slot: Option<u32>,
    live_names: Option<LiveNames>,
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
            live_names: None,
        })
    }

    /// Watch for voices the user names while the meeting runs.
    pub fn accept_names_from(&mut self, names: LiveNames) {
        self.live_names = Some(names);
    }

    /// Apply anything the user has named since the last utterance.
    fn take_live_names(&mut self) {
        let Some(queue) = &self.live_names else {
            return;
        };
        let pending: Vec<(u32, i64, String)> = {
            let Ok(mut queue) = queue.lock() else {
                return;
            };
            std::mem::take(&mut *queue)
        };

        for (slot_id, speaker_id, name) in pending {
            if let Some(slot) = self.slots.iter_mut().find(|slot| slot.id == slot_id) {
                tracing::debug!(slot = slot_id, %name, "voice named during the meeting");
                slot.known = Some((speaker_id, name));
            }
        }
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
        self.take_live_names();

        if duration < MIN_FOR_EVIDENCE {
            return self.carry_on();
        }

        let Some(embedding) = self.embed(samples) else {
            return self.carry_on();
        };

        // A confident match against a stored profile wins: the user already
        // told us who this is.
        let matched_profile = self
            .profiles
            .iter()
            .map(|profile| (profile, profile.similarity(&embedding)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, score)| *score >= self.thresholds.known)
            .map(|(profile, _)| (profile.speaker_id, profile.name.clone()));

        let best = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (index, cosine(&slot.centroid, &embedding)))
            .max_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((speaker_id, name)) = matched_profile {
            // Bind the name to a voice in this meeting, so later utterances
            // that fall short of the profile threshold still carry it.
            let slot = match best.filter(|(_, score)| *score >= self.thresholds.session) {
                Some((index, _)) => {
                    self.slots[index].absorb(&embedding, duration);
                    &mut self.slots[index]
                }
                None => {
                    let id = self.next_slot;
                    self.next_slot += 1;
                    self.slots.push(SessionSlot {
                        id,
                        centroid: embedding,
                        weight: duration.as_secs_f32(),
                        total_speech: duration,
                        utterances: 1,
                        known: None,
                    });
                    self.slots.last_mut().expect("just pushed")
                }
            };
            slot.known = Some((speaker_id, name.clone()));
            self.last_slot = Some(slot.id);
            return Attribution::Known { speaker_id, name };
        }

        if let Some((index, score)) = best
            && score >= self.thresholds.session
        {
            let slot = &mut self.slots[index];
            slot.absorb(&embedding, duration);
            self.last_slot = Some(slot.id);

            // A voice already recognised keeps its name even when this
            // particular utterance would not have matched the profile alone.
            return match &slot.known {
                Some((speaker_id, name)) => Attribution::Known {
                    speaker_id: *speaker_id,
                    name: name.clone(),
                },
                None => Attribution::Slot {
                    id: slot.id,
                    is_new: false,
                },
            };
        }

        if duration < MIN_FOR_NEW_SLOT {
            return self.carry_on();
        }

        let id = self.next_slot;
        self.next_slot += 1;
        self.slots.push(SessionSlot {
            id,
            centroid: embedding,
            weight: duration.as_secs_f32(),
            total_speech: duration,
            utterances: 1,
            known: None,
        });
        self.last_slot = Some(id);

        Attribution::Slot { id, is_new: true }
    }

    /// Attribute an utterance too short to judge to whoever holds the floor.
    ///
    /// If that voice has a name, the interjection carries it: a two-word "угу"
    /// from someone already identified should not appear under a speaker
    /// number.
    fn carry_on(&self) -> Attribution {
        let Some(id) = self.last_slot else {
            return Attribution::Unknown;
        };
        match self.slots.iter().find(|slot| slot.id == id) {
            Some(SessionSlot {
                known: Some((speaker_id, name)),
                ..
            }) => Attribution::Known {
                speaker_id: *speaker_id,
                name: name.clone(),
            },
            _ => Attribution::Continuation { id },
        }
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

/// Group utterances by voice, seeing the whole meeting at once.
///
/// The online tracker decides who is speaking with only the past to go on, so
/// an early utterance can open a slot that later evidence would have merged.
/// Given the recording afterwards, every embedding can be compared with every
/// other, which is strictly more information — and is why keeping the audio is
/// worth the disk.
///
/// Average linkage: a cluster is joined when the *mean* similarity to it clears
/// the threshold, so one unusually clear utterance cannot drag in a whole group.
pub fn cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    // Every utterance starts in its own cluster; merge the closest pair until
    // nothing is close enough.
    let mut clusters: Vec<Vec<usize>> = (0..embeddings.len()).map(|i| vec![i]).collect();

    loop {
        let mut best: Option<(usize, usize, f32)> = None;

        for a in 0..clusters.len() {
            for b in (a + 1)..clusters.len() {
                let score = average_linkage(&clusters[a], &clusters[b], embeddings);
                if score >= threshold && best.is_none_or(|(_, _, current)| score > current) {
                    best = Some((a, b, score));
                }
            }
        }

        let Some((a, b, _)) = best else { break };
        let merged = clusters.remove(b);
        clusters[a].extend(merged);
    }

    // Number clusters by when their first utterance happened, so "Спикер 1" is
    // whoever spoke first rather than an artefact of merge order.
    clusters.sort_by_key(|members| members.iter().copied().min().unwrap_or(usize::MAX));

    let mut assignment = vec![0usize; embeddings.len()];
    for (label, members) in clusters.iter().enumerate() {
        for member in members {
            assignment[*member] = label;
        }
    }
    assignment
}

fn average_linkage(a: &[usize], b: &[usize], embeddings: &[Vec<f32>]) -> f32 {
    let mut total = 0.0;
    for i in a {
        for j in b {
            total += cosine(&embeddings[*i], &embeddings[*j]);
        }
    }
    total / (a.len() * b.len()) as f32
}

#[cfg(test)]
mod cluster_tests {
    use super::*;

    /// Points around a centre, as embeddings of one voice would be.
    fn around(centre: &[f32], jitter: f32, count: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                centre
                    .iter()
                    .map(|value| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let noise = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                        value + noise * jitter
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn two_voices_become_two_clusters() {
        let mut embeddings = around(&[1.0, 0.0, 0.0], 0.05, 4, 1);
        embeddings.extend(around(&[0.0, 1.0, 0.0], 0.05, 3, 2));

        let labels = cluster(&embeddings, 0.7);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 2, "got {labels:?}");
        assert!(labels[..4].iter().all(|l| *l == labels[0]));
        assert!(labels[4..].iter().all(|l| *l == labels[4]));
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn labels_follow_who_spoke_first() {
        let mut embeddings = around(&[0.0, 1.0, 0.0], 0.05, 2, 3);
        embeddings.extend(around(&[1.0, 0.0, 0.0], 0.05, 2, 4));
        let labels = cluster(&embeddings, 0.7);
        assert_eq!(labels[0], 0, "the first utterance should be speaker 0");
    }

    #[test]
    fn one_voice_stays_one_cluster() {
        let embeddings = around(&[0.3, 0.9, 0.1], 0.03, 6, 5);
        let labels = cluster(&embeddings, 0.7);
        assert!(labels.iter().all(|l| *l == 0), "one voice split: {labels:?}");
    }

    #[test]
    fn an_impossible_threshold_keeps_everything_apart() {
        let embeddings = around(&[1.0, 0.0], 0.01, 4, 6);
        let labels = cluster(&embeddings, 1.01);
        assert_eq!(labels, vec![0, 1, 2, 3]);
    }

    #[test]
    fn nothing_in_nothing_out() {
        assert!(cluster(&[], 0.7).is_empty());
    }
}

/// One utterance's place in a recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Caller's identifier, handed back with the result.
    pub id: i64,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Re-group a finished meeting's utterances by voice, using the recording.
///
/// Spans too short to carry a voice are skipped rather than guessed at, so they
/// keep whatever attribution they already had. Slots are numbered from one, to
/// match what the transcript shows.
pub fn rediarize(
    tracker: &SpeakerTracker,
    audio: &[f32],
    spans: &[Span],
    threshold: f32,
) -> Vec<(i64, u32)> {
    let mut ids = Vec::new();
    let mut embeddings = Vec::new();

    for span in spans {
        if span.end_ms - span.start_ms < MIN_FOR_EVIDENCE.as_millis() as i64 {
            continue;
        }
        let Some(slice) = slice_ms(audio, span.start_ms, span.end_ms) else {
            continue;
        };
        if let Some(embedding) = tracker.embed(slice) {
            ids.push(span.id);
            embeddings.push(embedding);
        }
    }

    cluster(&embeddings, threshold)
        .into_iter()
        .zip(ids)
        .map(|(label, id)| (id, label as u32 + 1))
        .collect()
}

/// Mean embedding per cluster, for storing as each voice's signature.
pub fn centroids(assignments: &[(i64, u32)], embeddings: &[Vec<f32>]) -> Vec<(u32, Vec<f32>)> {
    let mut by_slot: std::collections::BTreeMap<u32, Vec<&Vec<f32>>> = Default::default();
    for ((_, slot), embedding) in assignments.iter().zip(embeddings) {
        by_slot.entry(*slot).or_default().push(embedding);
    }

    by_slot
        .into_iter()
        .filter_map(|(slot, members)| {
            let dim = members.first()?.len();
            let mut mean = vec![0.0f32; dim];
            for member in &members {
                for (value, incoming) in mean.iter_mut().zip(*member) {
                    *value += incoming;
                }
            }
            for value in &mut mean {
                *value /= members.len() as f32;
            }
            Some((slot, mean))
        })
        .collect()
}

fn slice_ms(audio: &[f32], start_ms: i64, end_ms: i64) -> Option<&[f32]> {
    let rate = ASR_SAMPLE_RATE as i64;
    let from = (start_ms.max(0) * rate / 1000) as usize;
    let to = ((end_ms.max(0) * rate / 1000) as usize).min(audio.len());
    (from < to).then(|| &audio[from..to])
}
