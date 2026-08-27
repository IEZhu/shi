//! Searching past meetings.

use shi_audio::StreamKind;
use shi_store::{MATCH_CLOSE, MATCH_OPEN, NewSegment, SessionSlot, Store, fts_query};

const START: &str = "2026-08-27T10:00:00+03:00[Europe/Moscow]";

fn seeded() -> Store {
    let store = Store::in_memory().expect("store");

    let lines = [
        ("Понедельник", "Выкатили платёжный шлюз, метрики ровные."),
        ("Понедельник", "Миграцию базы данных отложили до среды."),
        ("Вторник", "Шлюз держит нагрузку, откатывать не нужно."),
        ("Вторник", "Обсудили найм и бюджет на следующий квартал."),
    ];

    let mut current = String::new();
    let mut meeting_id = 0;
    for (title, text) in lines {
        if title != current {
            meeting_id = store
                .start_meeting(title, START, "parakeet-v3-int8")
                .expect("meeting")
                .id;
            current = title.to_string();
        }
        store
            .append_segment(
                meeting_id,
                &NewSegment {
                    stream: StreamKind::System,
                    start_ms: 0,
                    end_ms: 3_000,
                    speaker_id: None,
                    session_slot: Some(1),
                    text: text.into(),
                },
            )
            .expect("append");
    }
    store
}

#[test]
fn a_word_is_found_across_meetings() {
    let store = seeded();
    let hits = store.search("шлюз", 20).expect("search");

    assert_eq!(hits.len(), 2, "both meetings mention the gateway");
    let titles: Vec<&str> = hits.iter().map(|h| h.meeting_title.as_str()).collect();
    assert!(titles.contains(&"Понедельник"));
    assert!(titles.contains(&"Вторник"));
}

#[test]
fn matches_are_marked_without_producing_markup() {
    // Transcript text is whatever people said; handing the UI markup to parse
    // would turn a spoken tag into a rendering decision.
    let store = seeded();
    let hits = store.search("шлюз", 5).expect("search");
    let snippet = &hits[0].snippet;

    assert!(snippet.contains(MATCH_OPEN), "no match marker in {snippet:?}");
    assert!(snippet.contains(MATCH_CLOSE));
    assert!(!snippet.contains('<'), "snippet should carry no markup");
}

#[test]
fn search_is_case_insensitive_in_cyrillic() {
    let store = seeded();
    for query in ["ШЛЮЗ", "Шлюз", "шлюз"] {
        assert_eq!(
            store.search(query, 20).expect("search").len(),
            2,
            "{query} should match regardless of case"
        );
    }
}

#[test]
fn typing_finds_results_before_the_word_is_finished() {
    // The last term gets a prefix wildcard, so results appear as-you-type.
    let store = seeded();
    assert!(!store.search("мигр", 20).expect("search").is_empty());
    assert!(!store.search("бюдж", 20).expect("search").is_empty());
}

#[test]
fn several_words_narrow_rather_than_widen() {
    let store = seeded();
    let one = store.search("шлюз", 20).expect("search").len();
    let two = store.search("шлюз нагрузку", 20).expect("search").len();
    assert!(two < one, "adding a word should narrow: {two} vs {one}");
    assert_eq!(two, 1);
}

#[test]
fn punctuation_and_query_syntax_cannot_break_the_search() {
    // These are all valid things to type and invalid FTS5 syntax.
    let store = seeded();
    for query in [r#"шлюз" OR "#, "AND", "NOT шлюз", "*", "(((", "\"\"\""] {
        let result = store.search(query, 20);
        assert!(result.is_ok(), "{query:?} produced an error: {result:?}");
    }
}

#[test]
fn an_empty_query_returns_nothing_rather_than_everything() {
    let store = seeded();
    for query in ["", "   ", "\"\""] {
        assert!(
            store.search(query, 20).expect("search").is_empty(),
            "{query:?} should match nothing"
        );
        assert_eq!(fts_query(query), None);
    }
}

#[test]
fn the_index_follows_a_speaker_being_named() {
    // Naming a voice rewrites attribution, and search results carry the name,
    // so the two must not drift apart.
    let mut store = seeded();
    let meeting = store.meetings().expect("meetings").last().expect("first").id;
    store
        .upsert_session_slot(&SessionSlot {
            meeting_id: meeting,
            slot: 1,
            centroid: vec![0.1, 0.9],
            model_id: "titanet-small".into(),
            sample_path: None,
            total_speech_ms: 3_000,
            utterances: 1,
            resolved_speaker_id: None,
        })
        .expect("slot");
    store
        .name_session_slot(meeting, 1, "Мария", START)
        .expect("name");

    let hits = store.search("шлюз", 20).expect("search");
    assert!(
        hits.iter().any(|h| h.speaker_name.as_deref() == Some("Мария")),
        "a named speaker should show up in search results"
    );
}

#[test]
fn deleting_a_meeting_removes_it_from_the_index() {
    let store = seeded();
    let meeting = store.meetings().expect("meetings")[0].id;
    store.delete_meeting(meeting).expect("delete");

    let hits = store.search("шлюз", 20).expect("search");
    assert!(
        hits.iter().all(|h| h.meeting_id != meeting),
        "a deleted meeting still appears in search"
    );
}
