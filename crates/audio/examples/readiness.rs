//! Manual smoke test for M0: prove both capture streams are live.
//!
//! System audio cannot be verified with `cargo run` — macOS attributes the
//! capture request to the responsible process, which for a shell-launched
//! binary is the terminal, and an unauthorised tap returns silence rather than
//! an error. Run it the way the real app runs:
//!
//!     scripts/dev-run.sh readiness 15
//!
//! Talk into the microphone and play something through the speakers. The two
//! meters must move independently — that independence is what the whole
//! diarization design rests on.

use std::io::Write;
use std::time::{Duration, Instant};

use shi_audio::{AudioSource, MicSource, StreamHandle, SystemSource};

/// How often peaks are sampled, and how often a line is emitted.
const POLL: Duration = Duration::from_millis(100);
const REPORT_EVERY: Duration = Duration::from_millis(500);

fn bar(level: f32) -> String {
    let filled = (level.clamp(0.0, 1.0) * 40.0).round() as usize;
    format!("{}{}", "#".repeat(filled), "-".repeat(40 - filled))
}

/// Retarget stdout at a file. A bundled app has no terminal to print to.
fn redirect_stdout(path: &str) {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let Ok(file) = std::fs::File::create(path) else {
        return;
    };
    let fd = file.into_raw_fd();
    // SAFETY: `fd` is a freshly opened valid descriptor, and dup2 onto stdout
    // is the documented way to retarget it.
    unsafe {
        libc::dup2(fd, std::io::stdout().as_raw_fd());
        libc::close(fd);
    }
}

fn start(label: &str, source: &mut dyn AudioSource) -> Option<StreamHandle> {
    match source.start() {
        Ok(handle) => {
            println!(
                "{label:<8} started: {} @ {} Hz, {} ch",
                handle.info.device_name, handle.info.sample_rate, handle.info.channels
            );
            Some(handle)
        }
        Err(err) => {
            println!("{label:<8} FAILED: {err}");
            None
        }
    }
}

fn report(label: &str, handle: &StreamHandle) {
    let captured = handle.stats.frames_captured();
    let dropped = handle.stats.frames_dropped();
    let seconds = captured as f64 / handle.info.sample_rate.max(1) as f64;

    let verdict = if captured == 0 {
        "NO FRAMES — the device delivered nothing"
    } else if !handle.stats.has_signal() {
        "SILENT — frames arrive but every sample is zero"
    } else {
        "OK"
    };

    println!("{label:<8} {captured} frames ({seconds:.1} s), {dropped} dropped  ->  {verdict}");

    if captured > 0 && !handle.stats.has_signal() {
        println!("         macOS refuses system-audio capture by returning silence rather than");
        println!("         an error. Launch from a signed .app carrying NSAudioCaptureUsageDescription");
        println!("         (scripts/dev-run.sh does this); a bare CLI binary is attributed to the");
        println!("         terminal, which does not hold the grant.");
    }
}

fn main() {
    let to_file = std::env::var("DEV_LOG").ok();
    if let Some(path) = &to_file {
        redirect_stdout(path);
    }
    // Overwriting one line is nicer live, but unreadable in a log file.
    let line_end = if to_file.is_some() { '\n' } else { '\r' };

    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(15);

    let mut mic = MicSource::default_device();
    let mut system = SystemSource::new();

    let mic_handle = start("mic", &mut mic);
    let system_handle = start("system", &mut system);

    if mic_handle.is_none() && system_handle.is_none() {
        println!("\nneither stream started — nothing to measure");
        std::process::exit(1);
    }

    println!("\nspeak into the mic and play some audio; {seconds} s\n");

    let started = Instant::now();
    let mut last_report = Instant::now();
    let (mut mic_peak, mut sys_peak) = (0.0f32, 0.0f32);

    while started.elapsed() < Duration::from_secs(seconds) {
        std::thread::sleep(POLL);

        // take_peak resets, so accumulate across polls to keep the meter honest
        // when reporting less often than we sample.
        if let Some(h) = &mic_handle {
            mic_peak = mic_peak.max(h.stats.take_peak());
        }
        if let Some(h) = &system_handle {
            sys_peak = sys_peak.max(h.stats.take_peak());
        }

        if last_report.elapsed() >= REPORT_EVERY {
            print!("mic [{}]  sys [{}]{line_end}", bar(mic_peak), bar(sys_peak));
            let _ = std::io::stdout().flush();
            mic_peak = 0.0;
            sys_peak = 0.0;
            last_report = Instant::now();
        }
    }
    println!("\n");

    if let Some(h) = &mic_handle {
        report("mic", h);
    }
    if let Some(h) = &system_handle {
        report("system", h);
    }

    mic.stop();
    system.stop();
}
