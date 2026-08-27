use std::time::Duration;

/// Fraction of each interval we are willing to spend decoding draft text.
/// The rest is headroom for the final decodes, diarization and the UI.
const BUDGET: f32 = 0.5;

const BASE_INTERVAL: Duration = Duration::from_millis(800);
const MAX_INTERVAL: Duration = Duration::from_millis(4_000);
/// Below this a draft is too short to be worth showing.
const MIN_WINDOW: Duration = Duration::from_millis(2_000);
/// Above this the draft is long enough that the user has stopped reading it.
const MAX_WINDOW: Duration = Duration::from_millis(15_000);

/// Smoothing for the real-time factor estimate. Low enough to react within a
/// few decodes, high enough to ignore one slow scheduling hiccup.
const RTF_SMOOTHING: f32 = 0.3;

/// Decides how often to re-decode an in-progress utterance, and how much of it.
///
/// Draft text comes from re-running the same model over the still-open
/// utterance, which means the cost depends entirely on how fast that model
/// runs here. Published RTFx figures for Parakeet come from batched GPU runs
/// and say nothing about one CPU core, so this is a controller rather than a
/// constant: it measures what decoding actually costs and spends at most
/// [`BUDGET`] of each interval on it.
///
/// On a fast machine it settles at a short interval and a long window. On a
/// slow one the interval grows and the window shrinks, and drafts degrade
/// gradually instead of the pipeline falling behind.
#[derive(Debug, Clone)]
pub struct Cadence {
    interval: Duration,
    /// Decode seconds per second of audio — the inverse of RTFx. Starts
    /// pessimistic so a slow machine is never flooded before the first measurement.
    rtf: f32,
}

impl Default for Cadence {
    fn default() -> Self {
        Self {
            interval: BASE_INTERVAL,
            rtf: 0.1,
        }
    }
}

impl Cadence {
    /// How long to wait before producing the next draft.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Measured decode cost per second of audio.
    pub fn rtf(&self) -> f32 {
        self.rtf
    }

    /// The longest audio window affordable at the current interval.
    pub fn max_window(&self) -> Duration {
        Duration::from_secs_f32(self.affordable_window_secs().clamp(
            MIN_WINDOW.as_secs_f32(),
            MAX_WINDOW.as_secs_f32(),
        ))
    }

    /// Audio seconds decodable within the budget, before clamping.
    fn affordable_window_secs(&self) -> f32 {
        if self.rtf <= f32::EPSILON {
            return MAX_WINDOW.as_secs_f32();
        }
        BUDGET * self.interval.as_secs_f32() / self.rtf
    }

    /// Feed back what a decode actually cost, and re-tune.
    ///
    /// `audio` is how much audio was decoded, `decode` how long that took.
    pub fn observe(&mut self, audio: Duration, decode: Duration) {
        let audio_secs = audio.as_secs_f32();
        if audio_secs <= f32::EPSILON {
            return;
        }

        let measured = decode.as_secs_f32() / audio_secs;
        self.rtf = self.rtf * (1.0 - RTF_SMOOTHING) + measured * RTF_SMOOTHING;

        // Grow the interval only when even the shortest useful window will not
        // fit the budget; shrink it back once a long window is affordable again.
        if self.affordable_window_secs() < MIN_WINDOW.as_secs_f32() {
            self.interval = (self.interval * 2).min(MAX_INTERVAL);
        } else if self.interval > BASE_INTERVAL {
            let halved = self.interval / 2;
            let would_afford = BUDGET * halved.as_secs_f32() / self.rtf;
            if would_afford >= MIN_WINDOW.as_secs_f32() {
                self.interval = halved.max(BASE_INTERVAL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a machine where decoding costs `rtf` seconds per audio second.
    fn settle(rtf: f32, rounds: usize) -> Cadence {
        let mut cadence = Cadence::default();
        for _ in 0..rounds {
            let window = cadence.max_window();
            let decode = Duration::from_secs_f32(window.as_secs_f32() * rtf);
            cadence.observe(window, decode);
        }
        cadence
    }

    #[test]
    fn a_fast_machine_keeps_the_base_interval() {
        let cadence = settle(0.02, 20);
        assert_eq!(cadence.interval(), BASE_INTERVAL);
        assert_eq!(cadence.max_window(), MAX_WINDOW, "should afford the cap");
    }

    #[test]
    fn a_slow_machine_backs_off_instead_of_falling_behind() {
        // 0.5 s of decode per second of audio: RTFx of 2, painfully slow.
        let cadence = settle(0.5, 20);
        assert!(
            cadence.interval() > BASE_INTERVAL,
            "interval should grow, got {:?}",
            cadence.interval()
        );
        assert!(cadence.interval() <= MAX_INTERVAL);
    }

    #[test]
    fn the_budget_is_respected_once_settled() {
        // The controller converges to spending exactly the budget, so compare
        // with a tolerance that suits wall-clock times rather than f32::EPSILON.
        const SLACK_SECS: f32 = 0.001;

        for rtf in [0.02, 0.1, 0.25, 0.5] {
            let cadence = settle(rtf, 30);
            let cost = cadence.max_window().as_secs_f32() * rtf;
            let allowed = BUDGET * cadence.interval().as_secs_f32();
            assert!(
                cost <= allowed + SLACK_SECS,
                "rtf {rtf}: cost {cost:.3}s over budget {allowed:.3}s"
            );
        }
    }

    #[test]
    fn degradation_is_gradual_not_a_cliff() {
        // Each step slower must cost the user latency or window, never both
        // going to zero: drafts should thin out, not stop.
        let mut previous_quality = f32::MAX;
        for rtf in [0.02, 0.1, 0.25, 0.5] {
            let cadence = settle(rtf, 30);
            // Audio seconds of draft per wall second — how "live" it feels.
            let quality = cadence.max_window().as_secs_f32() / cadence.interval().as_secs_f32();
            assert!(quality > 0.9, "rtf {rtf}: drafts became useless ({quality:.2})");
            assert!(quality <= previous_quality + 0.01, "rtf {rtf}: not monotonic");
            previous_quality = quality;
        }
    }

    #[test]
    fn recovery_shrinks_the_interval_again() {
        let mut cadence = settle(0.5, 20);
        let backed_off = cadence.interval();
        assert!(backed_off > BASE_INTERVAL);

        // The machine frees up — say the meeting's other apps quit.
        for _ in 0..30 {
            let window = cadence.max_window();
            cadence.observe(window, Duration::from_secs_f32(window.as_secs_f32() * 0.02));
        }
        assert_eq!(cadence.interval(), BASE_INTERVAL, "should return to base");
    }

    #[test]
    fn a_zero_length_decode_is_ignored() {
        let mut cadence = Cadence::default();
        let before = cadence.rtf();
        cadence.observe(Duration::ZERO, Duration::from_millis(50));
        assert_eq!(cadence.rtf(), before, "must not divide by zero audio");
    }
}

