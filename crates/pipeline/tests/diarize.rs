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
