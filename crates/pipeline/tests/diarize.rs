//! Speaker tracking against real recorded voices.
//!
//! The fixtures deliberately include the hard case: two female voices in
//! closely related languages, which the calibration found to be the tightest
//! pair any of the candidate models had to separate.
//!
//! These need the embedding model, so they skip when it is absent; run
//! `scripts/fetch-models.sh` to enable them.

use std::path::{Path, PathBuf};
use std::time::Duration;

use shi_pipeline::{Attribution, SpeakerTracker, Thresholds, VoiceProfile};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn model() -> Option<String> {
    let path = repo_root().join("models/nemo_en_titanet_small.onnx");
    path.is_file().then(|| path.to_string_lossy().into_owned())
}

/// A fixture clip as 16 kHz mono, with its duration.
fn clip(name: &str) -> (Vec<f32>, Duration) {
    let path = repo_root().join("fixtures/voices").join(format!("{name}.wav"));
    let mut reader = hound::WavReader::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
    let rate = reader.spec().sample_rate;
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
        .collect();
    let duration = Duration::from_secs_f64(samples.len() as f64 / rate as f64);
    (samples, duration)
}

fn tracker() -> Option<SpeakerTracker> {
    let model = model()?;
    Some(SpeakerTracker::new(&model, 2, Thresholds::default()).expect("build tracker"))
}

fn slot_of(attribution: &Attribution) -> Option<u32> {
    match attribution {
        Attribution::Slot { id, .. } | Attribution::Continuation { id } => Some(*id),
        _ => None,
    }
}

#[test]
fn two_voices_get_two_slots_and_keep_them() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent — run scripts/fetch-models.sh");
        return;
    };

    // Interleaved, as a conversation actually arrives.
    let order = ["Milena_1", "Daniel_1", "Milena_2", "Daniel_2"];
    let slots: Vec<Option<u32>> = order
        .iter()
        .map(|name| {
            let (samples, duration) = clip(name);
            slot_of(&tracker.attribute(&samples, duration))
        })
        .collect();

    let milena = slots[0].expect("first utterance should open a slot");
    let daniel = slots[1].expect("second voice should open its own slot");

    assert_ne!(milena, daniel, "two speakers collapsed into one slot");
    assert_eq!(slots[2], Some(milena), "Milena was not recognised on return");
    assert_eq!(slots[3], Some(daniel), "Daniel was not recognised on return");
    assert_eq!(tracker.slots().len(), 2, "spurious slots were created");
}

#[test]
fn the_hardest_pair_is_still_separated() {
    // Two female voices, Russian and Ukrainian: the tightest pair in the
    // calibration corpus. If anything is going to be confused, it is this.
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (milena, milena_len) = clip("Milena_1");
    let (lesya, lesya_len) = clip("Lesya_1");
    let first = slot_of(&tracker.attribute(&milena, milena_len)).expect("slot");
    let second = slot_of(&tracker.attribute(&lesya, lesya_len)).expect("slot");

    assert_ne!(
        first, second,
        "the two closest voices in the corpus were merged; \
         the session threshold is too permissive"
    );
}

#[test]
fn a_named_voice_is_recognised_in_a_later_meeting() {
    // The whole point of storing voiceprints: name someone once, and they are
    // labelled automatically from then on.
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (enrolment, _) = clip("Milena_1");
    let embedding = tracker.embed(&enrolment).expect("embed");
    tracker.load_profiles(vec![VoiceProfile {
        speaker_id: 7,
        name: "Мария".into(),
        embeddings: vec![embedding],
    }]);

    // A different sentence from the same voice, as a later meeting would bring.
    let (later, later_len) = clip("Milena_2");
    match tracker.attribute(&later, later_len) {
        Attribution::Known { speaker_id, name } => {
            assert_eq!(speaker_id, 7);
            assert_eq!(name, "Мария");
        }
        other => panic!("stored voice was not recognised: {other:?}"),
    }
}

#[test]
fn a_stored_voice_does_not_claim_someone_else() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (enrolment, _) = clip("Milena_1");
    let embedding = tracker.embed(&enrolment).expect("embed");
    tracker.load_profiles(vec![VoiceProfile {
        speaker_id: 7,
        name: "Мария".into(),
        embeddings: vec![embedding],
    }]);

    // Lesya is the closest other voice there is, and must still not be Мария.
    let (lesya, lesya_len) = clip("Lesya_1");
    match tracker.attribute(&lesya, lesya_len) {
        Attribution::Known { name, .. } => {
            panic!("a different speaker was labelled as {name}")
        }
        _ => {}
    }
}

#[test]
fn a_short_interjection_continues_the_current_turn() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (samples, duration) = clip("Daniel_1");
    let opened = slot_of(&tracker.attribute(&samples, duration)).expect("slot");

    // "угу" — too short to identify anyone, and almost always the same person
    // still holding the floor.
    let brief = &samples[..(0.3 * 16_000.0) as usize];
    let attribution = tracker.attribute(brief, Duration::from_millis(300));

    assert_eq!(
        attribution,
        Attribution::Continuation { id: opened },
        "a short sound should neither create a speaker nor be dropped"
    );
    assert_eq!(tracker.slots().len(), 1, "an interjection created a speaker");
}

#[test]
fn nothing_is_attributed_before_anyone_has_spoken() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };
    let (samples, _) = clip("Daniel_1");
    let brief = &samples[..(0.2 * 16_000.0) as usize];
    assert_eq!(
        tracker.attribute(brief, Duration::from_millis(200)),
        Attribution::Unknown
    );
}

/// Build a "recording" by laying clips end to end, and the spans that describe
/// where each one sits — exactly the shape a finished meeting has.
fn recording(order: &[&str]) -> (Vec<f32>, Vec<shi_pipeline::Span>) {
    let mut audio: Vec<f32> = Vec::new();
    let mut spans = Vec::new();

    for (index, name) in order.iter().enumerate() {
        let (samples, duration) = clip(name);
        let start_ms = (audio.len() as f64 / 16.0) as i64;
        audio.extend_from_slice(&samples);
        spans.push(shi_pipeline::Span {
            id: index as i64,
            start_ms,
            end_ms: start_ms + duration.as_millis() as i64,
        });
        // A beat of silence between turns, as a real meeting has.
        audio.extend(std::iter::repeat_n(0.0, 16_000 / 2));
    }

    (audio, spans)
}

#[test]
fn reprocessing_a_recording_recovers_who_spoke() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent — run scripts/fetch-models.sh");
        return;
    };
    let _ = &mut tracker;

    let order = ["Milena_1", "Daniel_1", "Milena_2", "Daniel_2"];
    let (audio, spans) = recording(&order);

    let assignments = shi_pipeline::rediarize(
        &tracker,
        &audio,
        &spans,
        Some(shi_pipeline::DEFAULT_SESSION_THRESHOLD),
    );

    assert_eq!(assignments.len(), 4, "every utterance should be placed");
    let slots: Vec<u32> = assignments.iter().map(|(_, slot)| *slot).collect();

    assert_eq!(slots[0], slots[2], "Milena's two turns were split apart");
    assert_eq!(slots[1], slots[3], "Daniel's two turns were split apart");
    assert_ne!(slots[0], slots[1], "two speakers were merged into one");

    // Numbered by who spoke first, so the labels are stable and meaningful.
    assert_eq!(slots[0], 1);
    assert_eq!(slots[1], 2);
}

#[test]
fn reprocessing_skips_utterances_too_short_to_place() {
    let Some(tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (audio, mut spans) = recording(&["Milena_1", "Daniel_1"]);
    // A two-word interjection carries no usable voice.
    spans.push(shi_pipeline::Span {
        id: 99,
        start_ms: spans[0].start_ms,
        end_ms: spans[0].start_ms + 300,
    });

    let assignments = shi_pipeline::rediarize(
        &tracker,
        &audio,
        &spans,
        Some(shi_pipeline::DEFAULT_SESSION_THRESHOLD),
    );

    assert!(
        assignments.iter().all(|(id, _)| *id != 99),
        "a 300 ms span should keep its existing attribution, not be guessed at"
    );
    assert_eq!(assignments.len(), 2);
}

#[test]
fn a_recording_of_one_person_yields_one_voice() {
    let Some(tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (audio, spans) = recording(&["Daniel_1", "Daniel_2"]);
    let assignments = shi_pipeline::rediarize(
        &tracker,
        &audio,
        &spans,
        Some(shi_pipeline::DEFAULT_SESSION_THRESHOLD),
    );

    let slots: std::collections::HashSet<u32> =
        assignments.iter().map(|(_, slot)| *slot).collect();
    assert_eq!(slots.len(), 1, "one speaker was split into {slots:?}");
}

#[test]
fn a_recognised_voice_keeps_its_name_for_the_whole_meeting() {
    // Reported from a real run: the same person appeared as "Маша" on one line
    // and "Спикер 1" on the next. Utterances vary, and checking each one
    // against the stored profile independently lets some fall below the
    // threshold. The first confident match must bind the name to the voice.
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent — run scripts/fetch-models.sh");
        return;
    };

    let (enrolment, _) = clip("Milena_1");
    let embedding = tracker.embed(&enrolment).expect("embed");
    tracker.load_profiles(vec![VoiceProfile {
        speaker_id: 7,
        name: "Мария".into(),
        embeddings: vec![embedding],
    }]);

    // The enrolled utterance, then a different one from the same voice, then a
    // short interjection — every one of them should say Мария.
    let mut seen = Vec::new();
    for name in ["Milena_1", "Milena_2"] {
        let (samples, duration) = clip(name);
        seen.push(tracker.attribute(&samples, duration));
    }
    let (samples, _) = clip("Milena_1");
    seen.push(tracker.attribute(
        &samples[..(0.3 * 16_000.0) as usize],
        Duration::from_millis(300),
    ));

    for (index, attribution) in seen.iter().enumerate() {
        match attribution {
            Attribution::Known { name, speaker_id } => {
                assert_eq!(name, "Мария", "utterance {index}");
                assert_eq!(*speaker_id, 7);
            }
            other => panic!("utterance {index} lost the name: {other:?}"),
        }
    }
}

#[test]
fn binding_a_name_does_not_capture_a_different_voice() {
    // The other half: making identity sticky must not make it greedy.
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let (enrolment, _) = clip("Milena_1");
    let embedding = tracker.embed(&enrolment).expect("embed");
    tracker.load_profiles(vec![VoiceProfile {
        speaker_id: 7,
        name: "Мария".into(),
        embeddings: vec![embedding],
    }]);

    let (milena, milena_len) = clip("Milena_1");
    tracker.attribute(&milena, milena_len);

    // Lesya is the closest other voice in the corpus.
    let (lesya, lesya_len) = clip("Lesya_1");
    match tracker.attribute(&lesya, lesya_len) {
        Attribution::Known { name, .. } => panic!("a different speaker became {name}"),
        _ => {}
    }
}

#[test]
fn naming_a_voice_mid_meeting_applies_to_everything_said_after() {
    // Reported from a real run: the user named two voices during the call and
    // every later line still said "Спикер 1" and "Спикер 2". Naming writes to
    // the database and fixes the past, but the tracker runs on another thread
    // and never heard about it.
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent — run scripts/fetch-models.sh");
        return;
    };

    let names: shi_pipeline::LiveNames = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    tracker.accept_names_from(std::sync::Arc::clone(&names));

    // A voice speaks and opens a slot.
    let (first, first_len) = clip("Milena_1");
    let opened = match tracker.attribute(&first, first_len) {
        Attribution::Slot { id, .. } => id,
        other => panic!("expected a new voice, got {other:?}"),
    };

    // The user names it while the meeting is still running.
    names.lock().expect("lock").push((opened, 11, "Мария".into()));

    // Everything that voice says from here on carries the name.
    let (second, second_len) = clip("Milena_2");
    match tracker.attribute(&second, second_len) {
        Attribution::Known { speaker_id, name } => {
            assert_eq!(name, "Мария");
            assert_eq!(speaker_id, 11);
        }
        other => panic!("the name did not reach the running meeting: {other:?}"),
    }
}

#[test]
fn naming_one_voice_leaves_the_others_alone() {
    let Some(mut tracker) = tracker() else {
        eprintln!("skipping: speaker model absent");
        return;
    };

    let names: shi_pipeline::LiveNames = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    tracker.accept_names_from(std::sync::Arc::clone(&names));

    let (milena, milena_len) = clip("Milena_1");
    let milena_slot = match tracker.attribute(&milena, milena_len) {
        Attribution::Slot { id, .. } => id,
        other => panic!("expected a slot: {other:?}"),
    };
    let (daniel, daniel_len) = clip("Daniel_1");
    tracker.attribute(&daniel, daniel_len);

    names.lock().expect("lock").push((milena_slot, 11, "Мария".into()));

    let (daniel_again, daniel_again_len) = clip("Daniel_2");
    match tracker.attribute(&daniel_again, daniel_again_len) {
        Attribution::Known { name, .. } => panic!("an unnamed voice became {name}"),
        _ => {}
    }
}
