//! Does this segment hold a voice?
//!
//! The voice-activity detector answers a different question than its name
//! suggests: it finds where the signal is not silence. Over a real meeting it
//! opened 277 segments on the microphone stream, 49 of the 68 minutes, for a
//! person who spoke three times. What filled the rest was breath and room tone.
//!
//! Nothing downstream minded until the recogniser was made more willing to
//! speak, and then thirteen of those segments came back as "Thank you." and
//! "Yeah." in a meeting held in Russian. An empty transcript had been the only
//! thing catching them. `docs/transcription.md` has the full account.
//!
//! Loudness cannot be the test, however well it would have worked on that
//! recording. A faint talker sits exactly where a quiet room sits, and silencing
//! someone is a worse failure than printing a stray "Yeah."
//!
//! What separates them is that a voice repeats itself. Breath, a fan and a room
//! repeat themselves at no lag at all, however loud they are.
//!
//! What is measured is the longest unbroken stretch of that repetition, not the
//! share of the segment it covers. The share was the first attempt and it
//! dropped a hundred real words: a detector segment is mostly silence by
//! nature, so three seconds of speech inside a twenty-second segment scores
//! like noise. A stretch of voicing does not care what surrounds it.

/// Frame length for the pitch search: two periods of the lowest voice sought.
const FRAME: usize = 480; // 30 ms at 16 kHz

/// Step between frames. Fixed: a run measured in frames only means
/// something if every frame is the same length of time.
const HOP: usize = 160; // 10 ms

/// A human voice is periodic somewhere in here. Below is a hum from the room,
/// above is a whistle, and a meeting is made of neither.
const LOWEST_HZ: f32 = 70.0;
const HIGHEST_HZ: f32 = 400.0;

/// How strongly a frame must repeat itself to count as voiced.
const PERIODIC: f32 = 0.5;

/// Unbroken voicing below which a segment is not worth decoding.
///
/// Measured over one real meeting, both streams. The shortest run inside a
/// genuine utterance was 190 ms across 165 remote turns and 420 ms on the
/// microphone; the median run inside a segment that produced text out of
/// nothing was 80 ms, and inside a silent one 50 ms.
///
/// A hundred milliseconds is half the shortest real utterance seen and still
/// removes three quarters of the silent segments. The margin is deliberately on
/// this side: one meeting is thin evidence, and losing somebody's words is a
/// worse failure than printing a stray "Yeah."
pub const MIN_VOICED_MS: u32 = 100;

/// Shortest segment this can judge. Anything briefer is passed through: the
/// cost of being wrong that way is a stray line, and the other way it is
/// somebody's words.
pub const SHORTEST_JUDGED: usize = FRAME + (16_000.0 / LOWEST_HZ) as usize;

/// The longest unbroken stretch of voicing, in milliseconds.
///
/// Normalising the correlation by the energy of both windows is what makes the
/// answer independent of level: the same voice recorded faintly scores the same
/// as the voice recorded loudly.
///
/// `enough` stops the search early. A segment that has already shown enough
/// voice needs no further examination, which is the common case on a stream
/// carrying a conversation; pass `u32::MAX` to measure the whole thing.
pub fn voiced_run_ms(samples: &[f32], rate: u32, enough: u32) -> u32 {
    let shortest = (rate as f32 / HIGHEST_HZ) as usize;
    let longest = (rate as f32 / LOWEST_HZ) as usize;
    if samples.len() < FRAME + longest {
        return 0;
    }

    let per_frame = (HOP * 1000 / rate as usize) as u32;
    let (mut run, mut best) = (0u32, 0u32);
    let mut at = 0;
    while at + FRAME + longest <= samples.len() {
        if is_voiced(samples, at, shortest, longest) {
            run += per_frame;
            best = best.max(run);
            if best >= enough {
                return best;
            }
        } else {
            run = 0;
        }
        at += HOP;
    }
    best
}

/// Does the frame at `at` repeat itself at any lag a voice could have?
fn is_voiced(samples: &[f32], at: usize, shortest: usize, longest: usize) -> bool {
    let window = &samples[at..at + FRAME];
    let mean = window.iter().sum::<f32>() / FRAME as f32;

    for lag in shortest..=longest {
        let shifted = &samples[at + lag..at + lag + FRAME];
        let (mut dot, mut left, mut right) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..FRAME {
            let a = window[i] - mean;
            let b = shifted[i] - mean;
            dot += a * b;
            left += a * a;
            right += b * b;
        }
        let norm = (left * right).sqrt();
        if norm > f32::EPSILON && dot / norm >= PERIODIC {
            return true;
        }
    }
    false
}

/// Is this segment worth handing to a recogniser?
///
/// Fails open. A segment too short to measure is passed through, because the
/// cost of being wrong the other way is somebody's words going missing, and
/// this exists to stop stray text — not to police the transcript.
pub fn holds_speech(samples: &[f32], rate: u32) -> bool {
    holds_speech_beyond(samples, rate, MIN_VOICED_MS)
}

/// `holds_speech` with the bar stated, so a measurement can move it.
pub fn holds_speech_beyond(samples: &[f32], rate: u32, least_ms: u32) -> bool {
    if least_ms == 0 || samples.len() < SHORTEST_JUDGED {
        return true;
    }
    voiced_run_ms(samples, rate, least_ms) >= least_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    /// A pulse train: crude, but periodic in exactly the way a voice is.
    fn voice(hz: f32, seconds: f32, level: f32) -> Vec<f32> {
        let period = RATE as f32 / hz;
        (0..(RATE as f32 * seconds) as usize)
            .map(|n| {
                let phase = (n as f32 % period) / period;
                // A decaying pulse each period, which is what a glottal cycle
                // looks like from a distance.
                level * (-8.0 * phase).exp() * (phase * std::f32::consts::TAU * 3.0).sin()
            })
            .collect()
    }

    /// Deterministic pseudo-noise, so the test cannot fail on a Tuesday.
    fn hiss(seconds: f32, level: f32) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..(RATE as f32 * seconds) as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32 - 0.5) * 2.0 * level
            })
            .collect()
    }

    #[test]
    fn a_voice_is_voiced() {
        assert!(voiced_run_ms(&voice(120.0, 2.0, 0.2), RATE, u32::MAX) > 1_000);
    }

    #[test]
    fn room_noise_is_not() {
        assert!(voiced_run_ms(&hiss(2.0, 0.2), RATE, u32::MAX) < MIN_VOICED_MS);
    }

    #[test]
    fn silence_is_not() {
        assert_eq!(voiced_run_ms(&vec![0.0; RATE as usize * 2], RATE, u32::MAX), 0);
    }

    #[test]
    fn a_faint_voice_scores_the_same_as_a_loud_one() {
        // The whole reason this is not a loudness test. A hundredfold drop in
        // level is the difference between someone across the room and someone
        // at the microphone, and it must not be the difference between being
        // transcribed and being dropped.
        let loud = voiced_run_ms(&voice(120.0, 2.0, 0.5), RATE, u32::MAX);
        let faint = voiced_run_ms(&voice(120.0, 2.0, 0.005), RATE, u32::MAX);
        assert_eq!(loud, faint, "loud {loud} ms, faint {faint} ms");
        assert!(holds_speech(&voice(120.0, 2.0, 0.005), RATE));
    }

    #[test]
    fn loud_noise_is_still_not_speech() {
        // The mirror of the above: level alone must not buy a segment a decode.
        assert!(!holds_speech(&hiss(2.0, 0.9), RATE));
    }

    #[test]
    fn a_segment_too_short_to_judge_is_let_through() {
        assert!(holds_speech(&[0.0; 100], RATE));
    }

    #[test]
    fn a_voice_buried_in_noise_still_counts() {
        let voice = voice(140.0, 2.0, 0.05);
        let noise = hiss(2.0, 0.01);
        let mixed: Vec<f32> = voice.iter().zip(&noise).map(|(v, n)| v + n).collect();
        assert!(holds_speech(&mixed, RATE));
    }

    #[test]
    fn silence_around_the_speech_does_not_dilute_it() {
        // The failure that replaced the first design. A detector segment is
        // mostly silence by nature; a share-of-the-segment measure read three
        // seconds of speech inside twenty as noise and dropped a hundred real
        // words from one meeting.
        let mut padded = vec![0.0f32; RATE as usize * 8];
        padded.extend(voice(120.0, 1.0, 0.2));
        padded.extend(vec![0.0f32; RATE as usize * 8]);

        assert!(holds_speech(&padded, RATE), "buried in silence, but still spoken");
    }

    #[test]
    fn the_search_stops_once_it_has_seen_enough() {
        // Otherwise the check competes with the decode it exists to avoid.
        let long = voice(120.0, 20.0, 0.2);
        assert_eq!(voiced_run_ms(&long, RATE, MIN_VOICED_MS), MIN_VOICED_MS);
    }

    #[test]
    fn a_zero_bar_lets_everything_through() {
        assert!(holds_speech_beyond(&hiss(2.0, 0.2), RATE, 0));
    }
}
