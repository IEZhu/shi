//! The portability invariant.
//!
//! Every test here runs with no audio hardware whatsoever. If a test above
//! this layer ever needs a real device, a platform assumption has leaked into
//! code that must not have one — which is exactly what would make the Windows
//! and Linux ports a rewrite instead of a new `AudioSource`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use shi_audio::{AudioSource, FileSource, Pacing, StreamHandle, StreamKind};

/// Write an interleaved f32 WAV to a unique temp path.
fn write_wav(name: &str, samples: &[f32], channels: u16, sample_rate: u32) -> PathBuf {
    let path = std::env::temp_dir().join(format!("shi-{name}-{}.wav", std::process::id()));
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    for sample in samples {
        writer.write_sample(*sample).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    path
}

/// Drain exactly `expected` samples, or fail if they do not arrive in time.
fn drain(handle: &mut StreamHandle, expected: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(expected);
    let deadline = Instant::now() + Duration::from_secs(10);
    while out.len() < expected {
        match handle.consumer.pop() {
            Ok(sample) => out.push(sample),
            Err(_) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out after {} of {expected} samples",
                    out.len()
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
    out
}

#[test]
fn mono_replay_is_sample_exact_and_lossless() {
    let input: Vec<f32> = (0..16_000)
        .map(|i| (i as f32 / 16_000.0 * std::f32::consts::TAU * 440.0).sin() * 0.5)
        .collect();
    let path = write_wav("mono", &input, 1, 16_000);

    let mut source = FileSource::new(&path, StreamKind::System, Pacing::Immediate);
    let mut handle = source.start().expect("start file source");

    assert_eq!(handle.info.sample_rate, 16_000);
    assert_eq!(handle.info.channels, 1);
    assert_eq!(handle.info.kind, StreamKind::System);

    let got = drain(&mut handle, input.len());
    source.stop();

    assert_eq!(got.len(), input.len());
    for (i, (a, b)) in input.iter().zip(&got).enumerate() {
        assert!((a - b).abs() < 1e-6, "sample {i} differs: {a} vs {b}");
    }
    assert_eq!(
        handle.stats.frames_dropped(),
        0,
        "Immediate pacing must never drop; a lossy fixture makes assertions meaningless"
    );
    assert!(handle.stats.has_signal());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn stereo_is_downmixed_by_averaging() {
    // Hard-panned opposites: a correct downmix cancels to silence, while a
    // channel-picking or summing bug would show +0.5 or +1.0.
    let mut input = Vec::new();
    for _ in 0..1_000 {
        input.push(0.5);
        input.push(-0.5);
    }
    let path = write_wav("stereo", &input, 2, 16_000);

    let mut source = FileSource::new(&path, StreamKind::Mic, Pacing::Immediate);
    let mut handle = source.start().expect("start file source");
    assert_eq!(handle.info.channels, 2);

    let got = drain(&mut handle, input.len() / 2);
    source.stop();

    assert_eq!(got.len(), 1_000);
    for (i, sample) in got.iter().enumerate() {
        assert!(sample.abs() < 1e-6, "frame {i} should cancel, got {sample}");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn silence_never_sets_the_signal_flag() {
    // The whole point of has_signal: an authorised-but-silent stream and a
    // refused one look identical in frame counts, and only this tells them apart.
    let path = write_wav("silence", &vec![0.0f32; 4_000], 1, 16_000);

    let mut source = FileSource::new(&path, StreamKind::System, Pacing::Immediate);
    let mut handle = source.start().expect("start file source");
    let got = drain(&mut handle, 4_000);
    source.stop();

    assert_eq!(got.len(), 4_000);
    assert!(handle.stats.frames_captured() >= 4_000);
    assert!(
        !handle.stats.has_signal(),
        "digital silence must not count as signal"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn starting_twice_is_refused() {
    let path = write_wav("twice", &vec![0.1f32; 800], 1, 16_000);
    let mut source = FileSource::new(&path, StreamKind::Mic, Pacing::Immediate);
    let _first = source.start().expect("first start");
    assert!(source.start().is_err(), "a running source must refuse restart");
    source.stop();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_backlog_is_dropped_so_the_clock_can_start_clean() {
    // A source opened before the pipeline's clock starts fills its ring in the
    // meantime. Keeping that audio would stamp it at time zero and slide the
    // whole stream forward by however long the wait was.
    let input: Vec<f32> = (0..8_000).map(|i| (i as f32 / 100.0).sin() * 0.3).collect();
    let path = write_wav("backlog", &input, 1, 16_000);

    let mut source = FileSource::new(&path, StreamKind::Mic, Pacing::Immediate);
    let mut handle = source.start().expect("start");

    // Let the replay put something in the ring before the clock starts.
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle.stats.frames_captured() < 1_000 {
        assert!(Instant::now() < deadline, "fixture never reached the ring");
        std::thread::sleep(Duration::from_millis(1));
    }

    let dropped = handle.discard_backlog();
    assert!(dropped > 0, "nothing was waiting, so the test proves nothing");
    assert!(
        handle.consumer.pop().is_err() || dropped >= 1_000,
        "the ring still held the backlog after discarding it"
    );

    source.stop();
}
