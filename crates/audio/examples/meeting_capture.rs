//! Record a real meeting: both streams, raw, with no recognition running.
//!
//!     scripts/dev-run.sh meeting_capture
//!
//! Deliberately does nothing but drain the rings into two files. Recognition
//! during a live call costs CPU the call may want, and can be done afterwards
//! from the recording. Headerless 32-bit float PCM, so killing the process
//! costs nothing but the last few milliseconds — there is no header to
//! finalise and no way to lose what was already written.

use std::io::Write;
use std::time::{Duration, Instant};

use shi_audio::{AudioSource, MicSource, StreamHandle, SystemSource};

/// Status line interval.
const REPORT: Duration = Duration::from_secs(15);

fn redirect_stdout(path: &str) {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let Ok(file) = std::fs::File::create(path) else {
        return;
    };
    let fd = file.into_raw_fd();
    // SAFETY: freshly opened descriptor; dup2 onto stdout is the documented way.
    unsafe {
        libc::dup2(fd, std::io::stdout().as_raw_fd());
        libc::close(fd);
    }
}

struct Track {
    file: std::fs::File,
    frames: u64,
    peak: f32,
}

impl Track {
    fn create(base: &str, label: &str, handle: &StreamHandle) -> Option<Self> {
        let path = format!("{base}.{label}.f32");
        let file = std::fs::File::create(&path).ok()?;
        let _ = std::fs::write(
            format!("{path}.meta"),
            format!(
                "rate={} channels={} device={}\n",
                handle.info.sample_rate, handle.info.channels, handle.info.device_name
            ),
        );
        println!("{label:<7} -> {path}");
        Some(Self {
            file,
            frames: 0,
            peak: 0.0,
        })
    }

    fn drain(&mut self, handle: &mut StreamHandle, buffer: &mut Vec<u8>) {
        buffer.clear();
        while let Ok(sample) = handle.consumer.pop() {
            self.peak = self.peak.max(sample.abs());
            buffer.extend_from_slice(&sample.to_le_bytes());
            self.frames += 1;
        }
        if !buffer.is_empty() {
            let _ = self.file.write_all(buffer);
        }
    }

    fn report(&mut self, label: &str, rate: u32, stats: &shi_audio::StreamStats) {
        let seconds = self.frames as f64 / rate.max(1) as f64;
        println!(
            "  {label:<7} {seconds:7.1}s written, peak {:.3}, dropped {}{}",
            self.peak,
            stats.frames_dropped(),
            if stats.has_signal() { "" } else { "   <- NO SIGNAL YET" }
        );
        self.peak = 0.0;
    }
}

fn main() {
    let base = std::env::var("DEV_LOG").unwrap_or_else(|_| "meeting".into());
    redirect_stdout(&base);

    let mut mic_source = MicSource::default_device();
    let mut system_source = SystemSource::new();

    let mut mic = match mic_source.start() {
        Ok(handle) => handle,
        Err(err) => return println!("microphone did not start: {err}"),
    };
    let mut system = match system_source.start() {
        Ok(handle) => handle,
        Err(err) => return println!("system tap did not start: {err}"),
    };
    println!(
        "mic     {} @ {} Hz, {} ch\nsystem  {} @ {} Hz, {} ch",
        mic.info.device_name,
        mic.info.sample_rate,
        mic.info.channels,
        system.info.device_name,
        system.info.sample_rate,
        system.info.channels
    );

    let (Some(mut mic_track), Some(mut system_track)) = (
        Track::create(&base, "mic", &mic),
        Track::create(&base, "system", &system),
    ) else {
        return println!("could not open the output files");
    };

    // Same discipline as a meeting: drop what the rings collected before the
    // clock started, so the two files begin at the same moment.
    mic.discard_backlog();
    system.discard_backlog();
    let started = Instant::now();
    println!("\nrecording; kill the process to stop\n");

    let mut buffer = Vec::new();
    let mut last = Instant::now();
    loop {
        mic_track.drain(&mut mic, &mut buffer);
        system_track.drain(&mut system, &mut buffer);

        if last.elapsed() >= REPORT {
            println!("t+{:.0}s", started.elapsed().as_secs_f32());
            mic_track.report("mic", mic.info.sample_rate, &mic.stats);
            system_track.report("system", system.info.sample_rate, &system.stats);
            let _ = std::io::stdout().flush();
            last = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
