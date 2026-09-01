//! The dictionary end to end: an edit teaches a rule, and the rule repairs
//! every other line — including lines recorded before it was taught.

use shi_audio::StreamKind;
use shi_store::corrections;
use shi_store::markdown::MarkdownOptions;
use shi_store::{NewSegment, Store, markdown};

const START: &str = "2026-09-01T10:00:00+03:00";
const NOW: &str = "2026-09-01T11:00:00+03:00";

fn line(text: &str, at: i64) -> NewSegment {
    NewSegment {
        stream: StreamKind::System,
        start_ms: at,
        end_ms: at + 3_000,
        speaker_id: None,
        session_slot: None,
        text: text.into(),
    }
}

fn seeded() -> (Store, i64, Vec<i64>) {
    let store = Store::in_memory().expect("open store");
    let meeting = store
        .start_meeting("Разбор инцидента", START, "parakeet-v3-int8")
        .expect("start meeting")
        .id;
    let ids = [
        line("Мы посмотрели в Тыбаны, там всё красное.", 0),
        line("Квка не могла сделать запись.", 4_000),
        line("Тыбаны показывают тот же провал.", 8_000),
    ]
    .iter()
    .map(|segment| store.append_segment(meeting, segment).expect("append"))
    .collect();
    (store, meeting, ids)
}

#[test]
fn correcting_one_line_repairs_the_lines_around_it() {
    let (mut store, meeting, ids) = seeded();

    let learned = store
        .edit_segment(ids[0], "Мы посмотрели в Kibana, там всё красное.", NOW)
        .expect("edit");
    assert_eq!(learned, 1);

    let mut segments = store.segments(meeting).expect("segments");
    let fired = corrections::repair(&mut segments, &store.dictionary().expect("dictionary"));

    // The edited line was saved outright; the third line was never touched by
    // the user and is repaired by the rule the edit taught.
    assert_eq!(segments[0].text, "Мы посмотрели в Kibana, там всё красное.");
    assert_eq!(segments[2].text, "Kibana показывают тот же провал.");
    // The second line holds a word nobody has corrected yet.
    assert_eq!(segments[1].text, "Квка не могла сделать запись.");
    assert_eq!(fired.len(), 1);
}

#[test]
fn a_rule_learned_today_repairs_a_meeting_recorded_yesterday() {
    let (mut store, _, ids) = seeded();
    store
        .edit_segment(ids[0], "Мы посмотрели в Kibana, там всё красное.", NOW)
        .expect("edit");

    let older = store
        .start_meeting("Вчерашний созвон", "2026-08-31T10:00:00+03:00", "parakeet-v3-int8")
        .expect("start meeting")
        .id;
    store
        .append_segment(older, &line("В Тыбаны ничего не видно.", 0))
        .expect("append");

    let mut segments = store.segments(older).expect("segments");
    corrections::repair(&mut segments, &store.dictionary().expect("dictionary"));
    assert_eq!(segments[0].text, "В Kibana ничего не видно.");
}

#[test]
fn repaired_text_reaches_the_markdown() {
    let (mut store, meeting, ids) = seeded();
    store
        .edit_segment(ids[0], "Мы посмотрели в Kibana, там всё красное.", NOW)
        .expect("edit");

    let mut segments = store.segments(meeting).expect("segments");
    corrections::repair(&mut segments, &store.dictionary().expect("dictionary"));

    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &segments,
        &MarkdownOptions::default(),
    );
    assert!(rendered.contains("Kibana показывают тот же провал."), "{rendered}");
    assert!(!rendered.contains("Тыбаны"), "{rendered}");
}

#[test]
fn correcting_the_same_word_twice_keeps_the_newer_answer() {
    let (mut store, _, ids) = seeded();
    store
        .edit_segment(ids[0], "Мы посмотрели в Kibana, там всё красное.", NOW)
        .expect("first edit");
    store
        .edit_segment(ids[2], "Кибана показывают тот же провал.", NOW)
        .expect("second edit");

    let rules = store.corrections().expect("list");
    let kibana: Vec<_> = rules.iter().filter(|c| c.wrong == "тыбаны").collect();
    assert_eq!(kibana.len(), 1, "one rule per form, not one per edit");
    assert_eq!(kibana[0].right, "Кибана");
}

#[test]
fn a_rule_can_be_taught_directly_and_forgotten_again() {
    let (store, _, _) = seeded();
    store.teach("Квка", "Kafka", NOW).expect("teach");

    let rules = store.corrections().expect("list");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].wrong, "квка", "matched on folded letters");
    assert_eq!(rules[0].heard, "Квка", "displayed as it was written");
    assert_eq!(rules[0].right, "Kafka");

    store.forget_correction(rules[0].id).expect("forget");
    assert!(store.corrections().expect("list").is_empty());
    assert!(store.dictionary().expect("dictionary").is_empty());
}

#[test]
fn teaching_a_word_as_itself_is_refused() {
    let (store, _, _) = seeded();
    assert!(store.teach("Kafka", "kafka", NOW).is_err());
    assert!(store.teach("", "Kafka", NOW).is_err());
}

#[test]
fn applying_a_rule_counts_towards_it() {
    let (mut store, meeting, ids) = seeded();
    store
        .edit_segment(ids[0], "Мы посмотрели в Kibana, там всё красное.", NOW)
        .expect("edit");

    let mut segments = store.segments(meeting).expect("segments");
    let fired = corrections::repair(&mut segments, &store.dictionary().expect("dictionary"));
    store.count_hits(&fired).expect("count");

    let rules = store.corrections().expect("list");
    assert_eq!(rules[0].hits, 1, "one line still said Тыбаны");
}

#[test]
fn an_edit_that_teaches_nothing_still_saves_the_line() {
    let (mut store, meeting, ids) = seeded();
    let learned = store
        .edit_segment(ids[1], "Квка не могла сделать эту запись.", NOW)
        .expect("edit");

    assert_eq!(learned, 0, "an inserted word is not a vocabulary rule");
    let segments = store.segments(meeting).expect("segments");
    assert_eq!(segments[1].text, "Квка не могла сделать эту запись.");
    assert!(store.corrections().expect("list").is_empty());
}

#[test]
fn a_rule_survives_the_recogniser_splitting_the_word_differently() {
    // The same term arrived as "сред пул" on clean audio and "средпул"
    // through a telephone band. One correction has to cover both, or the
    // dictionary is only ever right about the recording it was taught on.
    let store = Store::in_memory().expect("open store");
    let meeting = store
        .start_meeting("Инцидент", START, "parakeet-v3-int8")
        .expect("start meeting")
        .id;
    let ids: Vec<i64> = [
        line("Сервис отдает таймаут, потому что сред пул забит.", 0),
        line("Опять средпул забит.", 4_000),
        line("И тут сред-пул тоже.", 8_000),
    ]
    .iter()
    .map(|segment| store.append_segment(meeting, segment).expect("append"))
    .collect();

    let mut store = store;
    store
        .edit_segment(
            ids[0],
            "Сервис отдает таймаут, потому что thread pool забит.",
            NOW,
        )
        .expect("edit");

    let mut segments = store.segments(meeting).expect("segments");
    corrections::repair(&mut segments, &store.dictionary().expect("dictionary"));
    assert_eq!(segments[1].text, "Опять thread pool забит.");
    assert_eq!(segments[2].text, "И тут thread pool тоже.");
}
