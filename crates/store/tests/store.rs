use shi_audio::StreamKind;
use shi_store::markdown::{MarkdownOptions, Timestamps};
use shi_store::{NewSegment, SessionSlot, Store, markdown};

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
            speaker_id: None,
            session_slot: None,
            text: "Выкатили ночью, метрики ровные.".into(),
        },
        NewSegment {
            stream: StreamKind::Mic,
            start_ms: 0,
            end_ms: 4_500,
            speaker_id: None,
            session_slot: None,
            text: "Доброе утро, давайте начнём со статусов.".into(),
        },
        NewSegment {
            stream: StreamKind::System,
            start_ms: 17_500,
            end_ms: 21_000,
            speaker_id: None,
            session_slot: None,
            text: "Откатывать не нужно.".into(),
        },
        NewSegment {
            stream: StreamKind::Mic,
            start_ms: 22_000,
            end_ms: 25_000,
            speaker_id: None,
            session_slot: None,
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
fn naming_a_voice_fixes_every_line_it_spoke_and_remembers_it() {
    // The core promise of identification: name a voice once, the whole
    // transcript updates, and the person is recognised in future meetings.
    let store = Store::in_memory().expect("open store");
    let meeting = store
        .start_meeting("Стендап", START, "parakeet-v3-int8")
        .expect("meeting");

    // Two utterances from the same unnamed voice, plus one from another.
    for (slot, start) in [(1u32, 0i64), (2, 6_000), (1, 12_000)] {
        store
            .append_segment(
                meeting.id,
                &NewSegment {
                    stream: StreamKind::System,
                    start_ms: start,
                    end_ms: start + 4_000,
                    speaker_id: None,
                    session_slot: Some(slot),
                    text: format!("реплика в {start}"),
                },
            )
            .expect("append");
    }

    store
        .upsert_session_slot(&SessionSlot {
            meeting_id: meeting.id,
            slot: 1,
            centroid: vec![0.1, 0.2, 0.3, 0.4],
            model_id: "titanet-small".into(),
            sample_path: None,
            total_speech_ms: 8_000,
            utterances: 2,
            resolved_speaker_id: None,
        })
        .expect("slot");

    let mut store = store;
    let speaker = store
        .name_session_slot(meeting.id, 1, "Мария", START)
        .expect("name the slot");

    let segments = store.segments(meeting.id).expect("segments");
    let named: Vec<_> = segments
        .iter()
        .filter(|s| s.speaker_name.as_deref() == Some("Мария"))
        .collect();
    assert_eq!(named.len(), 2, "both of that voice's lines should be claimed");
    assert!(
        segments
            .iter()
            .any(|s| s.session_slot == Some(2) && s.speaker_name.is_none()),
        "the other voice must be left alone"
    );

    // And the voice is now known, so the next meeting recognises it.
    let voices = store.voices_for_model("titanet-small").expect("voices");
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0].speaker_id, speaker.id);
    assert_eq!(voices[0].embeddings[0], vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn renaming_a_person_reaches_every_meeting_they_appear_in() {
    // Storing a reference rather than a copied name means this is one row,
    // not a rewrite of every transcript.
    let mut store = Store::in_memory().expect("store");

    let mut meetings = Vec::new();
    for title in ["Понедельник", "Вторник"] {
        let meeting = store
            .start_meeting(title, START, "parakeet-v3-int8")
            .expect("meeting");
        store
            .append_segment(
                meeting.id,
                &NewSegment {
                    stream: StreamKind::System,
                    start_ms: 0,
                    end_ms: 3_000,
                    speaker_id: None,
                    session_slot: Some(1),
                    text: "Привет.".into(),
                },
            )
            .expect("append");
        store
            .upsert_session_slot(&SessionSlot {
                meeting_id: meeting.id,
                slot: 1,
                centroid: vec![0.5, 0.5],
                model_id: "titanet-small".into(),
                sample_path: None,
                total_speech_ms: 3_000,
                utterances: 1,
                resolved_speaker_id: None,
            })
            .expect("slot");
        meetings.push(meeting.id);
    }

    let speaker = store
        .name_session_slot(meetings[0], 1, "Мария", START)
        .expect("name");
    // The same person turns up in the second meeting too.
    store
        .name_session_slot(meetings[1], 1, "Мария", START)
        .expect("name again");

    store
        .rename_speaker(speaker.id, "Мария Иванова")
        .expect("rename");

    for meeting in meetings {
        let segments = store.segments(meeting).expect("segments");
        assert_eq!(
            segments[0].speaker_name.as_deref(),
            Some("Мария Иванова"),
            "meeting {meeting} kept a stale name"
        );
    }
}

#[test]
fn embeddings_from_a_different_model_are_not_offered_for_matching() {
    // Cosine similarity between embeddings from different models is noise,
    // not evidence, so they must never be compared.
    let mut store = Store::in_memory().expect("store");
    let meeting = store
        .start_meeting("Стендап", START, "parakeet-v3-int8")
        .expect("meeting");
    store
        .upsert_session_slot(&SessionSlot {
            meeting_id: meeting.id,
            slot: 1,
            centroid: vec![1.0, 0.0],
            model_id: "campplus".into(),
            sample_path: None,
            total_speech_ms: 5_000,
            utterances: 1,
            resolved_speaker_id: None,
        })
        .expect("slot");
    store
        .name_session_slot(meeting.id, 1, "Мария", START)
        .expect("name");

    assert_eq!(store.voices_for_model("campplus").expect("voices").len(), 1);
    assert!(
        store
            .voices_for_model("titanet-small")
            .expect("voices")
            .is_empty(),
        "a voiceprint leaked across models"
    );
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
                speaker_id: None,
            session_slot: None,
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
fn a_named_voice_survives_reopening_the_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("meetings.db");

    let meeting_id = {
        let mut store = Store::open(&path).expect("open");
        let meeting = store
            .start_meeting("Планирование", START, "parakeet-v3-int8")
            .expect("meeting");
        store
            .append_segment(
                meeting.id,
                &NewSegment {
                    stream: StreamKind::System,
                    start_ms: 0,
                    end_ms: 2_000,
                    speaker_id: None,
                    session_slot: Some(1),
                    text: "Поехали.".into(),
                },
            )
            .expect("append");
        store
            .upsert_session_slot(&SessionSlot {
                meeting_id: meeting.id,
                slot: 1,
                centroid: vec![0.25, 0.75],
                model_id: "titanet-small".into(),
                sample_path: None,
                total_speech_ms: 2_000,
                utterances: 1,
                resolved_speaker_id: None,
            })
            .expect("slot");
        store
            .name_session_slot(meeting.id, 1, "Алексей", START)
            .expect("name");
        meeting.id
    };

    // Migrations must be safe to run again on an existing database.
    let store = Store::open(&path).expect("reopen");
    let segments = store.segments(meeting_id).expect("segments");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker_name.as_deref(), Some("Алексей"));
    assert_eq!(store.meetings().expect("meetings").len(), 1);
    assert_eq!(
        store.voices_for_model("titanet-small").expect("voices").len(),
        1,
        "the voiceprint must outlive the session that produced it"
    );
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
                speaker_id: None,
            session_slot: None,
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

#[test]
fn forgetting_a_voice_reverts_its_lines_rather_than_losing_them() {
    // A voiceprint enrolled under the wrong name will mislabel every future
    // meeting that person attends, so undoing it has to work — and it must
    // leave the transcript knowing those lines were still one voice.
    let mut store = Store::in_memory().expect("store");
    let meeting = store
        .start_meeting("Стендап", START, "parakeet-v3-int8")
        .expect("meeting");

    for start in [0i64, 5_000] {
        store
            .append_segment(
                meeting.id,
                &NewSegment {
                    stream: StreamKind::System,
                    start_ms: start,
                    end_ms: start + 3_000,
                    speaker_id: None,
                    session_slot: Some(1),
                    text: format!("реплика {start}"),
                },
            )
            .expect("append");
    }
    store
        .upsert_session_slot(&SessionSlot {
            meeting_id: meeting.id,
            slot: 1,
            centroid: vec![0.3, 0.7],
            model_id: "titanet-small".into(),
            sample_path: None,
            total_speech_ms: 6_000,
            utterances: 2,
            resolved_speaker_id: None,
        })
        .expect("slot");

    let wrong = store
        .name_session_slot(meeting.id, 1, "Ошибка", START)
        .expect("name");
    assert_eq!(store.voices_for_model("titanet-small").expect("voices").len(), 1);

    store.forget_speaker(wrong.id).expect("forget");

    assert!(
        store
            .voices_for_model("titanet-small")
            .expect("voices")
            .is_empty(),
        "the voiceprint must not survive; it would mislabel later meetings"
    );

    let segments = store.segments(meeting.id).expect("segments");
    assert!(
        segments.iter().all(|s| s.speaker_name.is_none()),
        "the wrong name lingered"
    );
    assert!(
        segments.iter().all(|s| s.session_slot == Some(1)),
        "attribution was lost entirely instead of reverting to the slot"
    );
}
