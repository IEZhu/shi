//! The catalogue has to be internally consistent, because a wrong entry only
//! shows up as a failed download much later.

use std::collections::HashSet;

use shi_models::{CATALOGUE, Family, Install, ModelKind, by_id, required};

#[test]
fn every_entry_is_uniquely_identified() {
    let ids: HashSet<&str> = CATALOGUE.iter().map(|spec| spec.id).collect();
    assert_eq!(ids.len(), CATALOGUE.len(), "duplicate model id");

    let paths: HashSet<&str> = CATALOGUE
        .iter()
        .map(|spec| match spec.install {
            Install::File(name) | Install::Archive(name) => name,
        })
        .collect();
    assert_eq!(
        paths.len(),
        CATALOGUE.len(),
        "two models would install to the same path"
    );
}

#[test]
fn checksums_are_well_formed() {
    for spec in CATALOGUE {
        let hash = spec.integrity.expected();
        assert_eq!(hash.len(), 64, "{}: not a sha-256", spec.id);
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "{}: checksum should be lowercase hex",
            spec.id
        );
    }
}

#[test]
fn whisper_entries_carry_the_prefix_their_files_use() {
    // sherpa ships Whisper as `turbo-encoder.int8.onnx` and friends, so a
    // missing prefix would produce paths that silently do not exist.
    for spec in CATALOGUE {
        match spec.family {
            Family::Whisper => assert!(
                spec.file_prefix.is_some(),
                "{}: Whisper needs a file prefix",
                spec.id
            ),
            _ => assert!(
                spec.file_prefix.is_none(),
                "{}: only Whisper uses a file prefix",
                spec.id
            ),
        }
    }
}

#[test]
fn only_recognizers_declare_a_recognizer_family() {
    for spec in CATALOGUE {
        let is_recognizer = spec.kind == ModelKind::Recognizer;
        let has_family = spec.family != Family::NotApplicable;
        assert_eq!(
            is_recognizer, has_family,
            "{}: family and kind disagree",
            spec.id
        );
    }
}

#[test]
fn a_default_set_covers_every_kind_a_meeting_needs() {
    let chosen = required(None);
    for kind in [
        ModelKind::Recognizer,
        ModelKind::Vad,
        ModelKind::SpeakerEmbedding,
    ] {
        assert!(
            chosen.iter().any(|spec| spec.kind == kind),
            "nothing default for {kind:?}; a fresh install could not start a meeting"
        );
    }
    assert_eq!(
        chosen.iter().filter(|s| s.kind == ModelKind::Recognizer).count(),
        1,
        "exactly one recogniser should be selected"
    );
}

#[test]
fn choosing_a_recognizer_replaces_the_default_rather_than_adding_to_it() {
    let chosen = required(Some("whisper-turbo"));
    let recognizers: Vec<&str> = chosen
        .iter()
        .filter(|s| s.kind == ModelKind::Recognizer)
        .map(|s| s.id)
        .collect();
    assert_eq!(recognizers, vec!["whisper-turbo"]);
}

#[test]
fn an_unknown_or_wrong_kind_choice_falls_back_to_the_default() {
    for choice in ["does-not-exist", "silero-vad"] {
        let chosen = required(Some(choice));
        let recognizer = chosen
            .iter()
            .find(|s| s.kind == ModelKind::Recognizer)
            .expect("a recogniser is always needed");
        assert!(
            recognizer.default,
            "{choice} should have fallen back to the default recogniser"
        );
    }
}

#[test]
fn lookup_finds_what_the_catalogue_advertises() {
    for spec in CATALOGUE {
        assert_eq!(by_id(spec.id).map(|s| s.id), Some(spec.id));
    }
    assert!(by_id("nothing-like-this").is_none());
}
