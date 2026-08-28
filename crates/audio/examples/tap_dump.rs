//! Dump the raw system-audio tap to a file so its content can be inspected.
//!
//!     scripts/dev-run.sh tap_dump 8
//!
//! Writes headerless 32-bit float PCM next to the log, at the tap's own rate,
//! plus a per-half-second frame timeline. Together they answer the two
//! questions a silent system stream raises: is the tap running at all, and is
//! what it delivers really the system mix?

use std::io::Write;
use std::time::{Duration, Instant};

use shi_audio::{AudioSource, SystemSource};

fn main() {
    let seconds: u64 = std::env::var("DUMP_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    // dev-run.sh hands every example a DEV_LOG path; park the dump beside it.
    let path = std::env::var("DEV_LOG")
        .map(|log| format!("{log}.f32"))
        .unwrap_or_else(|_| "tap.f32".into());

    let mut source = SystemSource::new();
    let mut handle = match source.start() {
        Ok(handle) => handle,
        Err(err) => {
            let _ = std::fs::write(format!("{path}.err"), format!("start failed: {err}\n"));
            return;
        }
    };

    let mut out = std::fs::File::create(&path).expect("create dump");
    let mut timeline = String::new();
    let mut last_tick = Instant::now();
    let mut last_frames = 0u64;
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(seconds) {
        if last_tick.elapsed() >= Duration::from_millis(500) {
            let now = handle.stats.frames_captured();
            timeline.push_str(&format!(
                "{:>5.1}s +{}\n",
                started.elapsed().as_secs_f32(),
                now - last_frames
            ));
            last_frames = now;
            last_tick = Instant::now();
        }
        let mut wrote = false;
        while let Ok(sample) = handle.consumer.pop() {
            let _ = out.write_all(&sample.to_le_bytes());
            wrote = true;
        }
        if !wrote {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    let _ = out.flush();
    let _ = std::fs::write(
        format!("{path}.meta"),
        format!(
            "rate={} channels={} device={} frames={} signal={}\n",
            handle.info.sample_rate,
            handle.info.channels,
            handle.info.device_name,
            handle.stats.frames_captured(),
            handle.stats.has_signal()
        ) + &timeline,
    );
    source.stop();
}
