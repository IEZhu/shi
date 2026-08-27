use std::sync::Arc;
use std::time::{Duration, Instant};

use sherpa_onnx::{
    LinearResampler, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use shi_audio::StreamKind;

use crate::asr::{ASR_SAMPLE_RATE, Transcriber};
use crate::cadence::Cadence;
use crate::diarize::{SessionSlot, SpeakerTracker};
use crate::echo::EchoReference;
use crate::error::{PipelineError, Result};
use crate::event::PipelineEvent;

/// Seconds of audio the VAD may buffer internally.
const VAD_BUFFER_SECONDS: f32 = 30.0;

/// Somewhere to keep the resampled audio, so the meeting can be re-processed
/// later with a better model.
///
/// A trait rather than a concrete writer because the pipeline has no business
/// knowing about file formats or retention policy; it only knows it has 16 kHz
/// mono samples that someone wants.
pub trait AudioTap: Send {
    fn write(&mut self, samples: &[f32]);
    /// Called once the stream ends. Consumes the tap so it cannot be written
    /// to afterwards.
    fn finish(self: Box<Self>);
}

/// What this stream does about acoustic echo.
///
/// Without headphones the microphone re-records whatever the speakers play, so
/// every remote utterance would be transcribed twice. The system stream
/// publishes what it hears; the microphone checks itself against it.
#[derive(Default)]
enum EchoRole {
    #[default]
    Ignore,
    Publish(Arc<EchoReference>),
    Suppress(Arc<EchoReference>),
}

/// Tuning for speech detection. `max_speech` is the important one: it forces a
/// long monologue to close so the transcript keeps flowing and re-decoding an
/// ever-growing draft cannot run away.
#[derive(Debug, Clone, Copy)]
pub struct VadSettings {
    pub threshold: f32,
    pub min_silence: Duration,
    pub min_speech: Duration,
    pub max_speech: Duration,
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            // Long enough to ride over the pause inside a sentence, short
            // enough that a turn ends promptly.
            min_silence: Duration::from_millis(500),
            // Below this it is a cough or a click, not a turn.
            min_speech: Duration::from_millis(250),
            max_speech: Duration::from_secs(20),
        }
    }
}

/// One capture stream's journey from samples to transcript events.
///
/// Both streams run an identical instance. The difference is downstream: what
/// the microphone produces belongs to one known speaker, while the system
/// stream's output goes on to diarization.
pub struct StreamPipeline {
    stream: StreamKind,
    resampler: Option<LinearResampler>,
    vad: VoiceActivityDetector,
    transcriber: Arc<dyn Transcriber>,
    cadence: Cadence,

    /// Audio at 16 kHz that no final has claimed yet, i.e. the utterance in
    /// progress. Drafts are decoded from this.
    open: Vec<f32>,
    /// Absolute index, in 16 kHz samples, of `open[0]`.
    open_start: u64,
    /// Total 16 kHz samples handed to the VAD, and the clock for timestamps.
    consumed: u64,
    last_draft: Option<Instant>,
    /// Whether a draft is currently on screen awaiting its final.
    draft_showing: bool,
    echo: EchoRole,
    suppressed: u64,
    /// Present only on the stream that carries several people.
    speakers: Option<SpeakerTracker>,
    recorder: Option<Box<dyn AudioTap>>,
}

impl StreamPipeline {
    pub fn new(
        stream: StreamKind,
        source_rate: u32,
        silero_model: &str,
        transcriber: Arc<dyn Transcriber>,
        settings: VadSettings,
    ) -> Result<Self> {
        // Sources rarely run at 16 kHz, so resample unless they happen to.
        let resampler = if source_rate as i32 == ASR_SAMPLE_RATE {
            None
        } else {
            Some(
                LinearResampler::create(source_rate as i32, ASR_SAMPLE_RATE).ok_or(
                    PipelineError::ResamplerCreateFailed {
                        from: source_rate,
                        to: ASR_SAMPLE_RATE as u32,
                    },
                )?,
            )
        };

        let mut config = VadModelConfig::default();
        config.silero_vad = SileroVadModelConfig {
            model: Some(silero_model.to_string()),
            threshold: settings.threshold,
            min_silence_duration: settings.min_silence.as_secs_f32(),
            min_speech_duration: settings.min_speech.as_secs_f32(),
            window_size: 512,
            max_speech_duration: settings.max_speech.as_secs_f32(),
        };
        config.sample_rate = ASR_SAMPLE_RATE;

        let vad = VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
            .ok_or(PipelineError::VadCreateFailed)?;

        Ok(Self {
            stream,
            resampler,
            vad,
            transcriber,
            cadence: Cadence::default(),
            open: Vec::new(),
            open_start: 0,
            consumed: 0,
            last_draft: None,
            draft_showing: false,
            echo: EchoRole::default(),
            suppressed: 0,
            speakers: None,
            recorder: None,
        })
    }

    /// Measured decode cost per second of audio, for diagnostics.
    pub fn rtf(&self) -> f32 {
        self.cadence.rtf()
    }

    /// Keep this stream's audio, so the meeting can be re-processed later.
    ///
    /// Recorded after resampling rather than at the device rate: what gets
    /// stored is then exactly what the models saw, so a later re-run cannot
    /// disagree with the original for reasons of resampling.
    pub fn record_to(&mut self, tap: Box<dyn AudioTap>) {
        self.recorder = Some(tap);
    }

    /// Attribute utterances on this stream to individual voices.
    ///
    /// Belongs on the system stream, which carries every remote participant
    /// mixed together. The microphone needs no tracker: it has one speaker by
    /// construction, which is the whole reason the streams are kept apart.
    pub fn identify_speakers(&mut self, tracker: SpeakerTracker) {
        self.speakers = Some(tracker);
    }

    /// Voices heard so far, for persisting between utterances.
    pub fn speaker_slots(&self) -> &[SessionSlot] {
        self.speakers.as_ref().map_or(&[], |t| t.slots())
    }

    /// Slots close enough to be one person the tracker split in two.
    pub fn merge_candidates(&self) -> Vec<(u32, u32, f32)> {
        self.speakers
            .as_ref()
            .map(|t| t.merge_candidates())
            .unwrap_or_default()
    }

    /// Publish this stream's audio so another can recognise it echoing back.
    /// Belongs on the system stream.
    pub fn publish_echo_reference(&mut self, reference: Arc<EchoReference>) {
        self.echo = EchoRole::Publish(reference);
    }

    /// Drop utterances that are `reference` arriving through the air. Belongs
    /// on the microphone stream.
    pub fn suppress_echo_of(&mut self, reference: Arc<EchoReference>) {
        self.echo = EchoRole::Suppress(reference);
    }

    /// Utterances discarded as echo.
    ///
    /// Surfaced rather than kept quiet: suppression deletes speech, so if it
    /// ever misfires the user needs to be able to see that it is happening
    /// instead of wondering why they are missing from their own transcript.
    pub fn suppressed_echo(&self) -> u64 {
        self.suppressed
    }

    /// Whether this utterance is the speakers coming back in through the mic.
    fn is_echo(&self, samples: &[f32], start: Duration) -> bool {
        match &self.echo {
            EchoRole::Suppress(reference) => reference.is_echo(samples, start),
            _ => false,
        }
    }

    /// Feed captured audio at the source's native rate and collect whatever
    /// the pipeline can say about it now.
    pub fn push(&mut self, native: &[f32]) -> Vec<PipelineEvent> {
        if native.is_empty() {
            return Vec::new();
        }

        let resampled = match &self.resampler {
            Some(r) => r.resample(native, false),
            None => native.to_vec(),
        };
        if resampled.is_empty() {
            return Vec::new();
        }

        if let Some(recorder) = self.recorder.as_mut() {
            recorder.write(&resampled);
        }

        if let EchoRole::Publish(reference) = &self.echo {
            reference.push(&resampled);
        }

        self.vad.accept_waveform(&resampled);
        self.open.extend_from_slice(&resampled);
        self.consumed += resampled.len() as u64;

        let mut events = self.collect_finals();
        if let Some(draft) = self.maybe_draft() {
            events.push(draft);
        }
        events
    }

    /// End of meeting: close any speech still buffered, and finish the
    /// recording.
    pub fn flush(&mut self) -> Vec<PipelineEvent> {
        if let Some(recorder) = self.recorder.take() {
            recorder.finish();
        }

        self.vad.flush();
        let mut events = self.collect_finals();
        if self.draft_showing {
            self.draft_showing = false;
            events.push(PipelineEvent::DraftAbandoned {
                stream: self.stream,
            });
        }
        events
    }

    /// Drain every utterance the VAD has closed.
    fn collect_finals(&mut self) -> Vec<PipelineEvent> {
        let mut events = Vec::new();

        while let Some(segment) = self.vad.front() {
            let start_sample = segment.start().max(0) as u64;
            let length = segment.n().max(0) as usize;
            let samples = segment.samples().to_vec();
            self.vad.pop();
            drop(segment);

            let start = samples_to_duration(start_sample);
            let end = samples_to_duration(start_sample + length as u64);

            // Check before decoding: an echo costs nothing to discard and a
            // decode is the most expensive thing this loop does.
            if self.is_echo(&samples, start) {
                self.suppressed += 1;
                tracing::debug!(
                    stream = %self.stream,
                    at = ?start,
                    total = self.suppressed,
                    "discarded an utterance as speaker echo"
                );
                self.discard_through(start_sample + length as u64);
                if self.draft_showing {
                    self.draft_showing = false;
                    events.push(PipelineEvent::DraftAbandoned {
                        stream: self.stream,
                    });
                }
                continue;
            }

            let began = Instant::now();
            let transcript = match self.transcriber.transcribe(&samples) {
                Ok(t) => t,
                Err(err) => {
                    tracing::warn!(stream = %self.stream, "final decode failed: {err}");
                    continue;
                }
            };
            self.cadence.observe(end.saturating_sub(start), began.elapsed());

            // Everything up to the end of this utterance is now accounted for.
            self.discard_through(start_sample + length as u64);

            if transcript.is_empty() {
                if self.draft_showing {
                    self.draft_showing = false;
                    events.push(PipelineEvent::DraftAbandoned {
                        stream: self.stream,
                    });
                }
                continue;
            }

            let speaker = self
                .speakers
                .as_mut()
                .map(|tracker| tracker.attribute(&samples, end.saturating_sub(start)));

            tracing::debug!(
                stream = %self.stream,
                at = ?start,
                len = ?end.saturating_sub(start),
                chars = transcript.text.chars().count(),
                ?speaker,
                "finalised an utterance"
            );

            self.draft_showing = false;
            events.push(PipelineEvent::Final {
                stream: self.stream,
                start,
                end,
                // Token offsets arrive relative to the utterance; shift them so
                // every timestamp in the transcript shares one origin.
                token_offsets: transcript
                    .token_offsets
                    .iter()
                    .map(|offset| start + *offset)
                    .collect(),
                text: transcript.text,
                tokens: transcript.tokens,
                speaker,
            });
        }

        events
    }

    /// Re-decode the utterance in progress, if speech is happening and the
    /// cadence controller says we can afford it.
    fn maybe_draft(&mut self) -> Option<PipelineEvent> {
        if !self.vad.detected() {
            return None;
        }

        let interval = self.cadence.interval();
        if self.last_draft.is_some_and(|at| at.elapsed() < interval) {
            return None;
        }

        // Decode only the affordable tail. Whatever precedes it has either been
        // finalised already or will be when the utterance closes.
        let window_samples = duration_to_samples(self.cadence.max_window());
        let offset = self.open.len().saturating_sub(window_samples);
        let window = &self.open[offset..];
        if window.is_empty() {
            return None;
        }

        // Suppress echoed drafts too, or the user watches their own transcript
        // fill with the other side's words and then empty again.
        let window_start = samples_to_duration(self.open_start + offset as u64);
        if self.is_echo(window, window_start) {
            self.last_draft = Some(Instant::now());
            return None;
        }

        let began = Instant::now();
        let transcript = self.transcriber.transcribe(window).ok()?;
        let elapsed = began.elapsed();
        self.cadence
            .observe(samples_to_duration(window.len() as u64), elapsed);
        self.last_draft = Some(Instant::now());

        if transcript.is_empty() {
            return None;
        }

        self.draft_showing = true;
        Some(PipelineEvent::Draft {
            stream: self.stream,
            start: window_start,
            text: transcript.text,
        })
    }

    /// Forget audio the transcript no longer needs.
    fn discard_through(&mut self, absolute_sample: u64) {
        let drop_count = absolute_sample.saturating_sub(self.open_start) as usize;
        if drop_count == 0 {
            return;
        }
        if drop_count >= self.open.len() {
            self.open.clear();
            self.open_start = absolute_sample;
        } else {
            self.open.drain(..drop_count);
            self.open_start = absolute_sample;
        }
    }
}

fn samples_to_duration(samples: u64) -> Duration {
    Duration::from_secs_f64(samples as f64 / ASR_SAMPLE_RATE as f64)
}

fn duration_to_samples(duration: Duration) -> usize {
    (duration.as_secs_f64() * ASR_SAMPLE_RATE as f64) as usize
}
