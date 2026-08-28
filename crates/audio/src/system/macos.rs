use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{AudioError, Result};
use crate::ring::RingWriter;
use crate::source::{AudioSource, SourceInfo, StreamHandle, StreamKind, ring_capacity};
use crate::stats::StreamStats;

#[repr(C)]
struct ShiTap {
    _private: [u8; 0],
}

type TapCallback =
    extern "C" fn(ctx: *mut c_void, samples: *const f32, frames: u32, channels: u32);

unsafe extern "C" {
    fn shi_tap_start(cb: TapCallback, ctx: *mut c_void, out_status: *mut i32) -> *mut ShiTap;
    fn shi_tap_stop(tap: *mut ShiTap);
    fn shi_tap_sample_rate(tap: *const ShiTap) -> u32;
    fn shi_tap_channels(tap: *const ShiTap) -> u32;
    fn shi_tap_device_name(tap: *const ShiTap, buf: *mut c_char, len: usize);
}

/// How many times to build the tap before giving up on it.
const START_ATTEMPTS: usize = 3;

/// How long a healthy tap may take to deliver its first frame. Measured at
/// roughly one IO cycle — 512 frames at 48 kHz, so ~11 ms — but device start-up
/// is not instant, and this only has to be short enough not to stall a meeting.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_millis(700);

/// Block until the tap proves it is actually running, or the timeout expires.
fn wait_for_first_frame(stats: &StreamStats) -> bool {
    let deadline = Instant::now() + FIRST_FRAME_TIMEOUT;
    while Instant::now() < deadline {
        if stats.frames_captured() > 0 {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    stats.frames_captured() > 0
}

/// Invoked on Core Audio's device IO thread. Realtime context.
extern "C" fn on_audio(ctx: *mut c_void, samples: *const f32, frames: u32, channels: u32) {
    if ctx.is_null() || samples.is_null() || frames == 0 {
        return;
    }
    let channels = channels.max(1);
    // SAFETY: `ctx` is the `RingWriter` leaked in `start` and kept alive until
    // `shi_tap_stop` returns, which is only called after the IO proc has been
    // stopped and destroyed. This thread is the only one touching it.
    let writer = unsafe { &mut *(ctx as *mut RingWriter) };
    let len = frames as usize * channels as usize;
    // SAFETY: Core Audio guarantees `samples` holds `frames * channels` floats.
    let block = unsafe { std::slice::from_raw_parts(samples, len) };
    writer.write_interleaved(block, channels as u16);
}

/// Captures the system audio mix through a Core Audio process tap.
///
/// This is the stream that carries every remote participant, and the only one
/// that needs diarization.
pub struct MacOsTapSource {
    tap: *mut ShiTap,
    writer: *mut RingWriter,
}

// The raw pointers are owned exclusively by this struct; Core Audio only
// touches them from its IO thread between `start` and `stop`.
unsafe impl Send for MacOsTapSource {}

impl MacOsTapSource {
    pub fn new() -> Self {
        Self {
            tap: std::ptr::null_mut(),
            writer: std::ptr::null_mut(),
        }
    }
}

impl Default for MacOsTapSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for MacOsTapSource {
    fn start(&mut self) -> Result<StreamHandle> {
        if !self.tap.is_null() {
            return Err(AudioError::AlreadyRunning);
        }

        // Chicken and egg: the callback needs a writer before the tap starts,
        // but the tap only reports its rate once running. Size the ring for the
        // highest rate we expect to see, which yields at least
        // DEFAULT_RING_SECONDS at any lower rate the device actually picks.
        const RING_RATE_HEADROOM: u32 = 48_000;
        let stats = Arc::new(StreamStats::default());
        let (producer, consumer) = rtrb::RingBuffer::new(ring_capacity(
            RING_RATE_HEADROOM,
            crate::source::DEFAULT_RING_SECONDS,
        ));
        let writer = Box::into_raw(Box::new(RingWriter::new(producer, Arc::clone(&stats))));

        let mut status: i32 = 0;
        let mut tap = std::ptr::null_mut();
        // A tap can start clean — every status noErr, the aggregate reporting
        // its sub-device active — and still never call its IO proc. Observed
        // on macOS 26.3 while building this; once frames do start they never
        // stop, so the failure belongs entirely to startup and a fresh tap
        // clears it. Refusing to hand back a stream that was never going to
        // deliver is the whole point: a meeting that records nothing is worse
        // than one that refuses to begin.
        for attempt in 0..START_ATTEMPTS {
            // SAFETY: `writer` stays valid until we call `shi_tap_stop` below.
            tap = unsafe { shi_tap_start(on_audio, writer as *mut c_void, &mut status) };
            if tap.is_null() {
                // SAFETY: the tap never started, so nothing else holds this pointer.
                drop(unsafe { Box::from_raw(writer) });
                return Err(AudioError::Platform {
                    context: "AudioHardwareCreateProcessTap",
                    status,
                });
            }

            // On the last attempt keep whatever we have. A tap that has not
            // delivered yet may still wake up when something plays, which is
            // how this worked before, and half a stream beats none. The
            // readiness panel reports `NoFrames` either way, so nobody is
            // told a lie about it.
            if wait_for_first_frame(&stats) || attempt + 1 == START_ATTEMPTS {
                break;
            }

            // SAFETY: `tap` came from `shi_tap_start` and is stopped once here.
            unsafe { shi_tap_stop(tap) };
            tap = std::ptr::null_mut();
        }

        // SAFETY: `tap` is non-null and owned by us.
        let sample_rate = unsafe { shi_tap_sample_rate(tap) };
        let channels = unsafe { shi_tap_channels(tap) } as u16;

        let mut name_buf = [0i8; 256];
        // SAFETY: buffer is 256 bytes and the shim NUL-terminates within it.
        unsafe { shi_tap_device_name(tap, name_buf.as_mut_ptr(), name_buf.len()) };
        let device_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        if sample_rate == 0 {
            unsafe { shi_tap_stop(tap) };
            drop(unsafe { Box::from_raw(writer) });
            return Err(AudioError::Unsupported(
                "system tap reported a zero sample rate".into(),
            ));
        }

        self.tap = tap;
        self.writer = writer;

        Ok(StreamHandle {
            info: SourceInfo {
                kind: StreamKind::System,
                device_name,
                sample_rate,
                channels: channels.max(1),
            },
            consumer,
            stats,
        })
    }

    fn stop(&mut self) {
        if self.tap.is_null() {
            return;
        }
        // Order matters: stopping the tap tears down the IO proc, so no further
        // callback can be in flight by the time we free the writer.
        // SAFETY: `self.tap` was returned by `shi_tap_start` and is stopped once.
        unsafe { shi_tap_stop(self.tap) };
        self.tap = std::ptr::null_mut();

        if !self.writer.is_null() {
            // SAFETY: callbacks have ceased, so we hold the only reference.
            drop(unsafe { Box::from_raw(self.writer) });
            self.writer = std::ptr::null_mut();
        }
    }

    fn kind(&self) -> StreamKind {
        StreamKind::System
    }
}

impl Drop for MacOsTapSource {
    fn drop(&mut self) {
        self.stop();
    }
}
