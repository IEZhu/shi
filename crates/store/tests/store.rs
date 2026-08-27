use shi_audio::StreamKind;
use shi_store::markdown::{MarkdownOptions, Timestamps};
use shi_store::{NewSegment, Store, markdown};

const START: &str = "2026-08-27T10:03:00+03:00";

fn seeded() -> (Store, i64) {
    let store = Store::in_memory().expect("open store");
    let meeting = store
        .start_meeting("Стендап команды", START, "parakeet-v3-int8")
        .expect("start meeting");

    // Deliberately inserted out of order and across both streams: the store is
    // responsible for reassembling one conversation from two capture streams.
    for segment in [
        NewSegment {
            stream: StreamKind::System,
            start_ms: 12_000,
            end_ms: 17_000,
            speaker: None,
            text: "Выкатили ночью, метрики ровные.".into(),
        },
        NewSegment {
            stream: StreamKind::Mic,
            start_ms: 0,
            end_ms: 4_500,
            speaker: None,
            text: "Доброе утро, давайте начнём со статусов.".into(),
        },
        NewSegment {
            stream: StreamKind::System,
            start_ms: 17_500,
            end_ms: 21_000,
            speaker: None,
            text: "Откатывать не нужно.".into(),
        },
        NewSegment {
            stream: StreamKind::Mic,
            start_ms: 22_000,
            end_ms: 25_000,
            speaker: None,
            text: "Отлично, спасибо.".into(),
        },
    ] {
        store.append_segment(meeting.id, &segment).expect("append");
    }

    (store, meeting.id)
}

#[test]
fn two_streams_reassemble_into_one_conversation() {
    let (store, meeting) = seeded();
    let segments = store.segments(meeting).expect("segments");

    let order: Vec<i64> = segments.iter().map(|s| s.start_ms).collect();
    assert_eq!(order, vec![0, 12_000, 17_500, 22_000], "not in spoken order");

    let streams: Vec<StreamKind> = segments.iter().map(|s| s.stream).collect();
    assert_eq!(
        streams,
        vec![
            StreamKind::Mic,
            StreamKind::System,
            StreamKind::System,
            StreamKind::Mic
        ],
        "stream attribution must survive the round trip"
    );
}

#[test]
fn naming_a_voice_afterwards_fixes_every_line_it_spoke() {
    // This is the reason Markdown is a projection rather than an append log.
    let (store, meeting) = seeded();

    let claimed = store
        .rename_speaker(meeting, None, "Мария")
        .expect("rename");
    assert_eq!(claimed, 4, "all unidentified segments should be claimed");

    let renamed = store
        .rename_speaker(meeting, Some("Мария"), "Мария Иванова")
        .expect("rename again");
    assert_eq!(renamed, 4);

    let segments = store.segments(meeting).expect("segments");
    assert!(
        segments
            .iter()
            .all(|s| s.speaker.as_deref() == Some("Мария Иванова")),
        "renaming must reach segments recorded before the name was known"
    );

    // And the rendered file follows, without touching it by hand.
    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &segments,
        &MarkdownOptions::default(),
    );
    assert!(rendered.contains("Мария Иванова"));
    assert!(!rendered.contains("Собеседник"), "stale label survived");
}

#[test]
fn unidentified_speakers_fall_back_to_their_stream() {
    let (store, meeting) = seeded();
    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &store.segments(meeting).expect("segments"),
        &MarkdownOptions::default(),
    );

    // Even with no diarization, the microphone is known to be the local user.
    assert!(rendered.contains("Вы:**"), "mic audio should be attributed");
    assert!(rendered.contains("Собеседник:**"));
}

#[test]
fn consecutive_turns_by_one_speaker_merge() {
    let (store, meeting) = seeded();
    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &store.segments(meeting).expect("segments"),
        &MarkdownOptions::default(),
    );

    // The two system utterances are adjacent, so they share one heading; an
    // hour of talk would be unreadable otherwise.
    assert_eq!(
        rendered.matches("Собеседник:**").count(),
        1,
        "adjacent turns should merge:\n{rendered}"
    );
    assert!(rendered.contains("Выкатили ночью, метрики ровные. Откатывать не нужно."));
    assert_eq!(rendered.matches("Вы:**").count(), 2, "separated turns stay apart");
}

#[test]
fn wall_clock_timestamps_track_the_meeting_start() {
    let (store, meeting) = seeded();
    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &store.segments(meeting).expect("segments"),
        &MarkdownOptions::default(),
    );

    assert!(rendered.contains("[10:03:00]"), "start offset 0\n{rendered}");
    assert!(rendered.contains("[10:03:12]"), "12 s in\n{rendered}");
    assert!(rendered.contains("[10:03:22]"), "22 s in\n{rendered}");
}

#[test]
fn relative_timestamps_are_offsets_from_zero() {
    let (store, meeting) = seeded();
    let rendered = markdown::render(
        &store.meeting(meeting).expect("meeting"),
        &store.segments(meeting).expect("segments"),
        &MarkdownOptions {
            timestamps: Timestamps::Relative,
            ..MarkdownOptions::default()
        },
    );

    assert!(rendered.contains("[00:00:00]"));
    assert!(rendered.contains("[00:00:12]"));
    assert!(!rendered.contains("[10:03"), "should not be wall clock");
}

#[test]
fn frontmatter_quotes_titles_that_would_break_yaml() {
    let store = Store::in_memory().expect("store");
    let meeting = store
        .start_meeting("Ретро: что пошло не так", START, "parakeet-v3-int8")
        .expect("meeting");
    store
        .append_segment(
            meeting.id,
            &NewSegment {
                stream: StreamKind::Mic,
                start_ms: 0,
                end_ms: 1_000,
                speaker: None,
                text: "Начнём.".into(),
            },
        )
        .expect("append");

    let rendered = markdown::render(
        &store.meeting(meeting.id).expect("meeting"),
        &store.segments(meeting.id).expect("segments"),
        &MarkdownOptions::default(),
    );
    assert!(
        rendered.contains(r#"title: "Ретро: что пошло не так""#),
        "a colon in the title must be quoted:\n{rendered}"
    );
}

#[test]
fn reopening_a_database_keeps_its_contents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("meetings.db");

    let meeting_id = {
        let store = Store::open(&path).expect("open");
        let meeting = store
            .start_meeting("Планирование", START, "parakeet-v3-int8")
            .expect("meeting");
        store
            .append_segment(
                meeting.id,
                &NewSegment {
                    stream: StreamKind::Mic,
                    start_ms: 0,
                    end_ms: 2_000,
                    speaker: Some("Алексей".into()),
                    text: "Поехали.".into(),
                },
            )
            .expect("append");
        meeting.id
    };

    // Migrations must be safe to run again on an existing database.
    let store = Store::open(&path).expect("reopen");
    let segments = store.segments(meeting_id).expect("segments");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker.as_deref(), Some("Алексей"));
    assert_eq!(store.meetings().expect("meetings").len(), 1);
}

#[test]
fn the_zone_annotated_format_survives_a_daylight_saving_boundary() {
    // The app records `Zoned::now()`, which carries the IANA zone. A fixed
    // numeric offset would drift across a DST change; the annotation does not.
    let store = Store::in_memory().expect("store");
    let meeting = store
        .start_meeting(
            "Ночная выкатка",
            "2026-10-25T02:30:00+03:00[Europe/Moscow]",
            "parakeet-v3-int8",
        )
        .expect("meeting");
    store
        .append_segment(
            meeting.id,
            &NewSegment {
                stream: StreamKind::Mic,
                start_ms: 0,
                end_ms: 1_000,
                speaker: None,
                text: "Начали.".into(),
            },
        )
        .expect("append");

    let rendered = markdown::render(
        &store.meeting(meeting.id).expect("meeting"),
        &store.segments(meeting.id).expect("segments"),
        &MarkdownOptions::default(),
    );
    assert!(rendered.contains("date: 2026-10-25"), "{rendered}");
    assert!(rendered.contains("[02:30:00]"), "{rendered}");
}
