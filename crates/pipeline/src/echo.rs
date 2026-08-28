use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use crate::asr::ASR_SAMPLE_RATE;

/// Envelope resolution. Fine enough to resolve a speaker-to-microphone delay,
/// coarse enough that correlating a long utterance is free.
const FRAME_MS: u64 = 10;
const FRAME_SAMPLES: usize = (ASR_SAMPLE_RATE as u64 * FRAME_MS / 1000) as usize;

/// How far apart the two streams may be and still be the same sound. Covers
/// the acoustic flight time, buffering in both capture paths, and a little
/// clock skew between two devices that are not synchronised to each other.
const MAX_LAG: Duration = Duration::from_millis(400);

/// Reference retained. Utterances are closed well inside this.
const RETAIN: Duration = Duration::from_secs(60);

/// Shortest overlap worth judging. Below half a second two envelopes can align
/// by chance, and the verdict deletes a line of transcript.
const MIN_OVERLAP: Duration = Duration::from_millis(500);

/// Correlation above which two envelopes are the same sound.
///
/// Independent speech lands well below this even when both people are talking
/// steadily; a loudspeaker echo of the very signal being played lands close to
/// one. Exposed because room, volume and microphone all move it.
pub const DEFAULT_ECHO_THRESHOLD: f32 = 0.7;

/// Loudness contour of the system stream, for the microphone to check itself
/// against.
///
/// Without headphones the microphone re-records whatever the speakers play, so
/// every remote utterance is transcribed twice — once from the digital tap and
/// once, worse, through the air. The two copies are the same sound, delayed.
///
/// Comparison is on short-term energy rather than raw samples deliberately: the
/// acoustic path filters, attenuates and delays the signal enough to destroy
/// sample-level correlation, while the loudness contour survives it. It is also
/// cheap — one number per 10 ms instead of per sample.
#[derive(Debug)]
pub struct EchoReference {
    inner: Mutex<Envelope>,
    threshold: f32,
}

#[derive(Debug, Default)]
struct Envelope {
    /// RMS per frame, oldest first.
    frames: VecDeque<f32>,
    /// Absolute frame index of `frames[0]`.
    first_frame: u64,
    /// Samples not yet forming a whole frame.
    partial: Vec<f32>,
}

impl EchoReference {
    pub fn new(threshold: f32) -> Self {
        Self {
            inner: Mutex::new(Envelope::default()),
            threshold,
        }
    }

    /// Feed system-stream audio, at 16 kHz, in capture order.
    pub fn push(&self, samples: &[f32]) {
        let mut envelope = lock(&self.inner);
        envelope.partial.extend_from_slice(samples);

        let whole = envelope.partial.len() / FRAME_SAMPLES;
        for index in 0..whole {
            let frame = &envelope.partial[index * FRAME_SAMPLES..(index + 1) * FRAME_SAMPLES];
            let energy = frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SAMPLES as f32;
            envelope.frames.push_back(energy.sqrt());
        }
        envelope.partial.drain(..whole * FRAME_SAMPLES);

        let keep = (RETAIN.as_millis() as u64 / FRAME_MS) as usize;
        while envelope.frames.len() > keep {
            envelope.frames.pop_front();
            envelope.first_frame += 1;
        }
    }

    /// How strongly a microphone utterance matches system audio around the same
    /// moment. Near 1.0 means the microphone is hearing the speakers.
    ///
    /// Returns `None` when there is no reference to compare against — early in
    /// a meeting, or when nothing is playing.
    pub fn similarity(&self, samples: &[f32], start: Duration) -> Option<f32> {
        let probe = envelope_of(samples);
        let min_overlap = frames_in(MIN_OVERLAP);
        if probe.len() < min_overlap {
            return None;
        }

        let lag = frames_in(MAX_LAG) as i64;
        let start_frame = (start.as_millis() as u64 / FRAME_MS) as i64;

        let envelope = lock(&self.inner);
        let first = envelope.first_frame as i64;
        let available = envelope.frames.len() as i64;

        // Widen by the lag in both directions: the echo trails the original,
        // but unsynchronised device clocks can nudge it either way.
        let from = (start_frame - lag).max(first);
        let to = (start_frame + probe.len() as i64 + lag).min(first + available);
        if to - from < min_overlap as i64 {
            return None;
        }

        let window: Vec<f32> = envelope
            .frames
            .iter()
            .skip((from - first) as usize)
            .take((to - from) as usize)
            .copied()
            .collect();
        drop(envelope);

        best_correlation(&probe, &window, start_frame - from, lag, min_overlap)
    }

    /// Whether an utterance is the speakers coming back in through the mic.
    pub fn is_echo(&self, samples: &[f32], start: Duration) -> bool {
        self.similarity(samples, start)
            .is_some_and(|score| score >= self.threshold)
    }

    /// The score at which an utterance is called an echo.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for EchoReference {
    fn default() -> Self {
        Self::new(DEFAULT_ECHO_THRESHOLD)
    }
}

/// RMS per frame.
fn envelope_of(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(FRAME_SAMPLES)
        .map(|frame| {
            (frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SAMPLES as f32).sqrt()
        })
        .collect()
}

/// Centre and scale to unit variance, so correlation ignores the fact that the
/// echo is quieter than the original.
fn normalise(values: &[f32]) -> Option<Vec<f32>> {
    if values.len() < 4 {
        return None;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let centred: Vec<f32> = values.iter().map(|v| v - mean).collect();
    let energy = centred.iter().map(|v| v * v).sum::<f32>().sqrt();
    if energy <= f32::EPSILON {
        // Digital silence has no contour to match.
        return None;
    }
    Some(centred.into_iter().map(|v| v / energy).collect())
}

/// Highest correlation between `probe` and `window` across the plausible lags.
///
/// The two are correlated over whatever they have in common rather than
/// requiring one to contain the other: a microphone utterance routinely runs
/// past the end of the remote audio that caused it, because the two streams
/// have their own voice-activity boundaries.
fn best_correlation(
    probe: &[f32],
    window: &[f32],
    nominal: i64,
    lag: i64,
    min_overlap: usize,
) -> Option<f32> {
    let mut best: Option<f32> = None;

    for shift in -lag..=lag {
        // Where probe[0] would sit inside the window at this alignment.
        let offset = nominal + shift;
        let low = offset.max(0);
        let high = (offset + probe.len() as i64).min(window.len() as i64);
        if high - low < min_overlap as i64 {
            continue;
        }

        // Normalise each overlap separately: scaling the whole probe against a
        // window it only partly covers would compare different quantities.
        let (Some(a), Some(b)) = (
            normalise(&probe[(low - offset) as usize..(high - offset) as usize]),
            normalise(&window[low as usize..high as usize]),
        ) else {
            continue;
        };

        let score: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        best = Some(best.map_or(score, |current: f32| current.max(score)));
    }

    best
}

fn frames_in(duration: Duration) -> usize {
    (duration.as_millis() as u64 / FRAME_MS) as usize
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise, so a failure is always reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// Speech-like audio: noise shaped by a syllable-rate envelope. What makes
    /// one utterance distinguishable from another is that contour, which is
    /// exactly what the detector compares.
    fn speech(seconds: f32, seed: u64, syllable_hz: f32) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let count = (seconds * ASR_SAMPLE_RATE as f32) as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / ASR_SAMPLE_RATE as f32;
                let syllable = (t * syllable_hz * std::f32::consts::TAU).sin().max(0.0);
                let pause = if (t * 0.7) as i32 % 3 == 2 { 0.05 } else { 1.0 };
                rng.next() * syllable * pause * 0.4
            })
            .collect()
    }

    /// What a loudspeaker and a room do to a signal before the mic hears it:
    /// delay, attenuate, dull the top end, and add a little room noise.
    fn through_the_air(source: &[f32], delay: Duration, gain: f32) -> Vec<f32> {
        let delay_samples = (delay.as_secs_f32() * ASR_SAMPLE_RATE as f32) as usize;
        let mut rng = Lcg(A1R_SEED);
        let mut low_passed = 0.0f32;

        let mut out = vec![0.0f32; delay_samples];
        for sample in source {
            low_passed = low_passed * 0.7 + sample * 0.3;
            out.push(low_passed * gain + rng.next() * 0.004);
        }
        out
    }

    const A1R_SEED: u64 = 0x5EED;

    #[test]
    fn a_loudspeaker_echo_is_recognised() {
        let played = speech(4.0, 1, 4.0);
        let reference = EchoReference::default();
        reference.push(&played);

        // The microphone hears it 120 ms later, quieter and duller.
        let heard = through_the_air(&played, Duration::from_millis(120), 0.25);
        let score = reference
            .similarity(&heard, Duration::ZERO)
            .expect("comparable");

        assert!(
            score >= DEFAULT_ECHO_THRESHOLD,
            "echo scored {score:.3}, below the {DEFAULT_ECHO_THRESHOLD} threshold"
        );
        assert!(reference.is_echo(&heard, Duration::ZERO));
    }

    #[test]
    fn the_local_speaker_talking_is_not_echo() {
        // Someone else is on the call while the user says something different.
        let remote = speech(4.0, 1, 4.0);
        let reference = EchoReference::default();
        reference.push(&remote);

        let local = speech(4.0, 99, 5.5);
        let score = reference
            .similarity(&local, Duration::ZERO)
            .expect("comparable");

        assert!(
            score < DEFAULT_ECHO_THRESHOLD,
            "independent speech scored {score:.3}, would be suppressed as echo"
        );
        assert!(!reference.is_echo(&local, Duration::ZERO));
    }

    #[test]
    fn headphones_leave_nothing_to_match_against() {
        // Nothing coming out of the speakers means an empty reference, and the
        // detector must stay out of the way rather than guess.
        let reference = EchoReference::default();
        let local = speech(3.0, 7, 4.5);
        assert_eq!(reference.similarity(&local, Duration::ZERO), None);
        assert!(!reference.is_echo(&local, Duration::ZERO));
    }

    #[test]
    fn silence_is_never_called_echo() {
        let reference = EchoReference::default();
        reference.push(&vec![0.0f32; ASR_SAMPLE_RATE as usize * 3]);
        let quiet = vec![0.0f32; ASR_SAMPLE_RATE as usize];
        assert_eq!(reference.similarity(&quiet, Duration::ZERO), None);
        assert!(!reference.is_echo(&quiet, Duration::ZERO));
    }

    #[test]
    fn an_utterance_from_a_different_moment_does_not_match() {
        // The search window is deliberately narrow: audio that resembles
        // something said thirty seconds ago is not an echo of it.
        let reference = EchoReference::default();
        reference.push(&speech(4.0, 1, 4.0));
        reference.push(&speech(4.0, 42, 6.0));

        let early = speech(2.0, 1, 4.0);
        let score = reference.similarity(&early, Duration::from_secs(6));
        assert!(
            score.is_none_or(|s| s < DEFAULT_ECHO_THRESHOLD),
            "matched across a six second gap: {score:?}"
        );
    }

    #[test]
    fn the_two_cases_are_separated_by_a_real_margin() {
        // A threshold that only just works is a threshold that will not survive
        // a different room. Assert the gap, not merely which side of it we land.
        let played = speech(4.0, 1, 4.0);
        let reference = EchoReference::default();
        reference.push(&played);

        let echo = reference
            .similarity(
                &through_the_air(&played, Duration::from_millis(120), 0.25),
                Duration::ZERO,
            )
            .expect("echo comparable");
        let independent = reference
            .similarity(&speech(4.0, 99, 5.5), Duration::ZERO)
            .expect("independent comparable");

        assert!(
            echo - independent > 0.3,
            "margin too thin: echo {echo:.3} vs independent {independent:.3}"
        );
        assert!(
            independent < DEFAULT_ECHO_THRESHOLD && DEFAULT_ECHO_THRESHOLD < echo,
            "threshold {DEFAULT_ECHO_THRESHOLD} sits outside the gap \
             ({independent:.3}..{echo:.3})"
        );
    }

    #[test]
    fn a_longer_utterance_than_the_reference_still_matches() {
        // The microphone's voice-activity boundaries are its own, so its
        // utterance routinely runs past the remote audio that caused it.
        let played = speech(2.0, 5, 4.0);
        let reference = EchoReference::default();
        reference.push(&played);

        let mut heard = through_the_air(&played, Duration::from_millis(90), 0.3);
        heard.extend(speech(1.5, 77, 5.0));

        assert!(
            reference.is_echo(&heard, Duration::ZERO),
            "an utterance overhanging the reference should still be caught"
        );
    }

    #[test]
    fn old_reference_audio_is_forgotten() {
        let reference = EchoReference::default();
        // Push more than the retention window and confirm memory is bounded.
        for _ in 0..80 {
            reference.push(&speech(1.0, 3, 4.0));
        }
        let frames = lock(&reference.inner).frames.len();
        let cap = (RETAIN.as_millis() as u64 / FRAME_MS) as usize;
        assert!(frames <= cap, "kept {frames} frames, cap is {cap}");
    }
}
