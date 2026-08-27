//! Storing meeting audio.
//!
//! Audio is kept so a meeting can be re-diarized or re-transcribed later, which
//! only makes sense if what comes back is what went in. These tests decode the
//! result rather than trusting the word "lossless".

use shi_audio::StreamKind;
use shi_store::{AudioStore, RECORDING_SAMPLE_RATE};

/// Speech-like audio: a tone with an amplitude contour, which compresses far
/// less trivially than silence would.
fn samples(seconds: f32) -> Vec<f32> {
    let count = (seconds * RECORDING_SAMPLE_RATE as f32) as usize;
    (0..count)
        .map(|i| {
            let t = i as f32 / RECORDING_SAMPLE_RATE as f32;
            let envelope = (t * 4.0 * std::f32::consts::TAU).sin().max(0.0);
            (t * 220.0 * std::f32::consts::TAU).sin() * envelope * 0.6
        })
        .collect()
}

/// Decode a FLAC file back to i16.
fn decode(path: &std::path::Path) -> Vec<i16> {
    let mut reader = claxon::FlacReader::open(path).expect("open flac");
    reader
        .samples()
        .map(|s| s.expect("sample") as i16)
        .collect()
}

#[test]
fn a_recording_is_compressed_losslessly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    let input = samples(2.0);
    let mut recorder = store.recorder(1, StreamKind::System).expect("recorder");
    // Written in blocks, the way the pipeline delivers it.
    for block in input.chunks(1024) {
        recorder.write(block);
    }
    let flac = recorder.finish().expect("finish");

    assert!(flac.is_file(), "no flac produced");
    assert_eq!(flac.extension().unwrap(), "flac");
    assert!(
        !flac.with_extension("wav").exists(),
        "the uncompressed original should be gone"
    );

    let decoded = decode(&flac);
    assert_eq!(decoded.len(), input.len(), "sample count changed");

    // The only permitted difference is the f32 -> i16 quantisation done on the
    // way in; the codec itself must add nothing.
    for (index, (original, roundtripped)) in input.iter().zip(&decoded).enumerate() {
        let expected = (original.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        assert_eq!(
            *roundtripped, expected,
            "sample {index} changed: {roundtripped} vs {expected}"
        );
    }
}

#[test]
fn compression_actually_saves_space() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    let input = samples(3.0);
    let uncompressed = input.len() as u64 * 2; // 16-bit mono

    let mut recorder = store.recorder(1, StreamKind::Mic).expect("recorder");
    recorder.write(&input);
    let flac = recorder.finish().expect("finish");

    let compressed = std::fs::metadata(&flac).expect("metadata").len();
    assert!(
        compressed < uncompressed,
        "flac ({compressed}) is not smaller than raw ({uncompressed})"
    );
}

#[test]
fn audio_survives_a_hard_kill_and_is_compressed_on_the_next_launch() {
    // The real failure this guards against: a recording interrupted with no
    // chance to close the file. An earlier design wrote WAV, whose header
    // records the data length only on close — so an interrupted recording
    // claimed zero samples and the recovery pass deleted it. Simulate a kill
    // by leaking the recorder so nothing runs on the way out.
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    let input = samples(2.0);
    {
        let mut recorder = store.recorder(7, StreamKind::System).expect("recorder");
        recorder.write(&input);
        // No finish, no drop: exactly what kill -9 leaves behind.
        std::mem::forget(recorder);
    }

    let raw = store.meeting_dir(7).join("system.pcm");
    let written = std::fs::metadata(&raw).expect("interrupted recording").len();
    assert!(written > 0, "nothing reached the file at all");

    assert_eq!(store.compress_orphans(), 1);
    assert!(!raw.exists(), "the raw file should have been replaced");

    let flac = store.meeting_dir(7).join("system.flac");
    assert!(flac.is_file(), "the recovered audio was not compressed");

    // What matters is that the audio is still there, not merely that a file is.
    let decoded = decode(&flac);
    assert_eq!(
        decoded.len(),
        (written / 2) as usize,
        "recovered sample count does not match what reached the disk"
    );
    assert!(
        decoded.len() as f32 > input.len() as f32 * 0.5,
        "most of the recording was lost: {} of {}",
        decoded.len(),
        input.len()
    );
    for (index, decoded_sample) in decoded.iter().enumerate() {
        let expected = (input[index].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        assert_eq!(*decoded_sample, expected, "sample {index} changed");
    }

    // And running the sweep again finds nothing to do.
    assert_eq!(store.compress_orphans(), 0);
}

#[test]
fn a_recording_that_never_captured_anything_leaves_nothing_behind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    let recorder = store.recorder(9, StreamKind::Mic).expect("recorder");
    recorder.finish().expect("finish");

    assert!(store.files_for(9).is_empty(), "an empty recording left a file");
}

#[test]
fn pruning_frees_space_and_reports_how_much() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    for meeting in [1, 2] {
        let mut recorder = store.recorder(meeting, StreamKind::System).expect("recorder");
        recorder.write(&samples(1.0));
        recorder.finish().expect("finish");
    }

    let before = store.usage_bytes();
    assert!(before > 0);

    let freed = store.prune(&[1]);
    assert!(freed > 0, "pruning reported nothing freed");
    assert!(!store.meeting_dir(1).exists());
    assert!(store.meeting_dir(2).exists(), "an unrelated meeting was removed");
    assert_eq!(store.usage_bytes(), before - freed);
}

#[test]
fn pruning_a_meeting_with_no_audio_is_harmless() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());
    assert_eq!(store.prune(&[42]), 0);
    assert_eq!(store.usage_bytes(), 0);
}

#[test]
fn both_streams_are_kept_apart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = AudioStore::new(dir.path());

    for stream in [StreamKind::Mic, StreamKind::System] {
        let mut recorder = store.recorder(3, stream).expect("recorder");
        recorder.write(&samples(0.5));
        recorder.finish().expect("finish");
    }

    let mut names: Vec<String> = store
        .files_for(3)
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    assert_eq!(names, vec!["mic.flac", "system.flac"]);
}
