//! What happens to an utterance between the microphone and the recogniser.
//!
//! Recognisers are trained on speech at a sane level with the rumble taken
//! out. Meeting audio is neither: the first real recording made with this
//! application had a microphone stream sitting at an RMS of 0.001 with peaks
//! at full scale, which is forty decibels below what a model expects to hear.
//!
//! Every stage here is optional and independently measurable, because a
//! preprocessing step that is not measured is a preprocessing step that
//! quietly makes things worse.

use std::path::Path;

use sherpa_onnx::{
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserGtcrnModelConfig,
    OfflineSpeechDenoiserModelConfig,
};

use crate::asr::ASR_SAMPLE_RATE;
use crate::error::{PipelineError, Result};

/// Level the normaliser aims for, as RMS of the whole utterance.
///
/// 0.08 is roughly -22 dBFS, which is where broadcast speech sits and where
/// the training corpora of these models mostly sit too.
pub const TARGET_RMS: f32 = 0.08;

/// Ceiling on how much an utterance may be amplified.
///
/// Without it a pause between words becomes a wall of hiss: the quieter the
/// input, the more eagerly the gain would climb.
///
/// Twenty decibels, and the first number here was wrong in an instructive way.
///
/// It was set to forty from the per-minute RMS of the first real recording,
/// which ran between 0.001 and 0.009. But those minutes were mostly silence:
/// measured per *utterance*, the two moments that speaker actually spoke sat at
/// 0.10 and 0.16 — above the target, needing no gain at all. Everything else
/// was breath and room at 0.00006 to 0.008, and forty decibels lifted it far
/// enough for the recogniser to write "Thank you." and "I'm just gonna be able
/// to do" over it. Thirteen invented lines, no recovered ones.
///
/// Twenty still rescues genuinely faint speech — an utterance at 0.008 reaches
/// the target — while leaving noise at 0.0001 twenty times below anything a
/// recogniser will hallucinate from.
pub const MAX_GAIN_DB: f32 = 20.0;

/// Highest sample allowed after gain, leaving headroom rather than clipping.
const CEILING: f32 = 0.97;

/// Remove any constant offset.
///
/// A DC offset costs headroom and shifts every frame's energy, which matters
/// to a voice-activity detector even when it is inaudible.
pub fn remove_dc(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    if mean.abs() > f32::EPSILON {
        for sample in samples.iter_mut() {
            *sample -= mean;
        }
    }
}

/// Root mean square of an utterance.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Bring an utterance to a standard loudness, and report the gain applied.
///
/// Returns 1.0 when the audio is already right, or when it is so quiet that
/// lifting it would only amplify the room.
pub fn normalize(samples: &mut [f32]) -> f32 {
    normalize_within(samples, MAX_GAIN_DB)
}

/// Normalise, but never amplify by more than `max_gain_db`.
///
/// The ceiling is the whole safety mechanism, so it is a parameter that can be
/// measured rather than a constant that has to be believed.
pub fn normalize_within(samples: &mut [f32], max_gain_db: f32) -> f32 {
    let level = rms(samples);
    if level <= f32::EPSILON {
        return 1.0;
    }

    let ceiling_gain = 10f32.powf(max_gain_db / 20.0);
    let mut gain = (TARGET_RMS / level).min(ceiling_gain);

    // Never let the loudest sample clip: a transient that survives the gain is
    // a click the recogniser has to explain away.
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak * gain > CEILING {
        gain = CEILING / peak;
    }
    if (gain - 1.0).abs() < 1e-3 {
        return 1.0;
    }
    for sample in samples.iter_mut() {
        *sample *= gain;
    }
    gain
}

/// A neural denoiser, applied to a whole utterance.
pub struct Denoiser {
    inner: OfflineSpeechDenoiser,
}

impl Denoiser {
    /// Load a GTCRN denoiser from a single `.onnx` file.
    pub fn load(model: impl AsRef<Path>, threads: i32) -> Result<Self> {
        let model = model.as_ref();
        if !model.is_file() {
            return Err(PipelineError::ModelFileMissing {
                label: "denoiser",
                path: model.to_path_buf(),
            });
        }
        let config = OfflineSpeechDenoiserConfig {
            model: OfflineSpeechDenoiserModelConfig {
                gtcrn: OfflineSpeechDenoiserGtcrnModelConfig {
                    model: Some(model.to_string_lossy().into_owned()),
                },
                num_threads: threads.max(1),
                ..OfflineSpeechDenoiserModelConfig::default()
            },
        };
        let inner =
            OfflineSpeechDenoiser::create(&config).ok_or(PipelineError::RecognizerCreateFailed)?;
        Ok(Self { inner })
    }

    /// Denoise in place, leaving the audio untouched if the model returns
    /// nothing usable.
    ///
    /// The utterance keeps its length whatever the model returns: timestamps
    /// are derived from it, and a denoiser is not allowed to move them.
    pub fn run(&self, samples: &mut [f32]) {
        let cleaned = self.inner.run(samples, ASR_SAMPLE_RATE as i32).samples;
        if cleaned.is_empty() {
            return;
        }
        let overlap = cleaned.len().min(samples.len());
        samples[..overlap].copy_from_slice(&cleaned[..overlap]);
    }
}

/// Which stages to run, and in what strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    pub remove_dc: bool,
    pub normalize: bool,
    pub denoise: bool,
}

impl Default for Settings {
    /// Levelling is off.
    ///
    /// It is a large win on genuinely faint speech — 43 % word error against
    /// 24 % on a corpus scaled down to a fortieth of normal — and a net loss on
    /// the first real meeting recorded here, where the quiet stretches were
    /// silence rather than faint speech. Levelled, the recogniser wrote
    /// "Thank you." and "I'm just gonna be able to do" over breath it had
    /// previously returned nothing for: thirteen invented lines, no recovered
    /// ones. Lowering the gain ceiling did not help — twenty decibels produced
    /// twenty-five such lines where forty produced twenty-three — because the
    /// ceiling is not the mechanism. The voice-activity detector hands over
    /// segments that hold no speech, and an empty transcript was the only thing
    /// catching them.
    ///
    /// So it is available and off, until there is a way to tell a faint talker
    /// from a quiet room.
    fn default() -> Self {
        Self {
            remove_dc: true,
            normalize: false,
            denoise: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, amplitude: f32) -> Vec<f32> {
        let count = (seconds * ASR_SAMPLE_RATE as f32) as usize;
        (0..count)
            .map(|i| {
                let t = i as f32 / ASR_SAMPLE_RATE as f32;
                (t * 220.0 * std::f32::consts::TAU).sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn room_noise_is_not_lifted_into_speech_range() {
        // Measured from a real meeting: between utterances the microphone sat
        // here, and a ceiling generous enough to reach the target from this
        // level is a ceiling that invents words out of breath.
        let mut room = tone(1.0, 0.0001);
        normalize(&mut room);
        assert!(
            rms(&room) < TARGET_RMS / 4.0,
            "room noise reached {:.4}, close enough to speech to be transcribed",
            rms(&room)
        );
    }

    #[test]
    fn a_quiet_utterance_is_brought_up_to_level() {
        // The first real recording sat here: an RMS of about a thousandth of
        // full scale.
        let mut quiet = tone(1.0, 0.012);
        let gain = normalize(&mut quiet);
        assert!(gain > 1.0, "quiet audio was left alone");
        assert!(gain < 10f32.powf(MAX_GAIN_DB / 20.0), "this case should not need the cap");
        assert!(
            (rms(&quiet) - TARGET_RMS).abs() < 0.02,
            "landed at {:.3} instead of {TARGET_RMS}",
            rms(&quiet)
        );
    }

    #[test]
    fn audio_already_at_level_is_left_alone() {
        let mut fine = tone(1.0, TARGET_RMS * std::f32::consts::SQRT_2);
        let before = fine.clone();
        assert_eq!(normalize(&mut fine), 1.0);
        assert_eq!(fine, before, "audio at the target level was rescaled anyway");
    }

    #[test]
    fn silence_is_not_amplified_into_noise() {
        let mut silence = vec![0.0f32; ASR_SAMPLE_RATE as usize];
        assert_eq!(normalize(&mut silence), 1.0);
        assert!(silence.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn the_gain_is_capped_so_a_whisper_does_not_become_a_hiss() {
        let mut almost_nothing = tone(1.0, 1e-5);
        let gain = normalize(&mut almost_nothing);
        let cap = 10f32.powf(MAX_GAIN_DB / 20.0);
        assert!(gain <= cap * 1.001, "gain {gain} exceeded the {cap} cap");
    }

    #[test]
    fn normalising_never_clips() {
        // A quiet utterance with one loud transient in it: the gain that suits
        // the speech would push the transient past full scale.
        let mut audio = tone(1.0, 0.005);
        audio[100] = 0.9;
        normalize(&mut audio);
        let peak = audio.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 1.0, "peak reached {peak}");
    }

    #[test]
    fn a_constant_offset_is_removed() {
        let mut shifted: Vec<f32> = tone(0.5, 0.1).iter().map(|s| s + 0.2).collect();
        remove_dc(&mut shifted);
        let mean = shifted.iter().sum::<f32>() / shifted.len() as f32;
        assert!(mean.abs() < 1e-4, "offset of {mean} survived");
    }
}
