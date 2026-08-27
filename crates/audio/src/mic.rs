use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, FromSample, SampleFormat, Stream};

use crate::error::{AudioError, Result};
use crate::ring::RingWriter;
use crate::source::{AudioSource, SourceInfo, StreamHandle, StreamKind, ring_capacity};
use crate::stats::StreamStats;

/// Scratch space for converting non-f32 device formats without allocating in
/// the callback. Blocks larger than this are converted in several passes.
const CONVERT_SCRATCH: usize = 8192;

/// Microphone capture via cpal.
///
/// This one implementation covers macOS, Windows and Linux — only the system
/// output stream needs per-OS code.
pub struct MicSource {
    device_name: Option<String>,
    stream: Option<Stream>,
}

impl MicSource {
    /// Capture from the OS default input device.
    pub fn default_device() -> Self {
        Self {
            device_name: None,
            stream: None,
        }
    }

    /// Capture from a device selected by name in settings.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            device_name: Some(name.into()),
            stream: None,
        }
    }
}

/// Map cpal's error kinds onto ours, keeping "you must grant access" separate
/// from "the device broke" — the readiness panel reacts differently to each.
fn map_cpal_error(err: cpal::Error) -> AudioError {
    match err.kind() {
        ErrorKind::PermissionDenied => AudioError::PermissionDenied("microphone"),
        ErrorKind::DeviceNotAvailable | ErrorKind::HostUnavailable => {
            AudioError::NoDevice("microphone")
        }
        ErrorKind::UnsupportedConfig => AudioError::Unsupported(err.to_string()),
        _ => AudioError::Device(err.to_string()),
    }
}

/// Build the data callback for a device format that is not already f32.
fn converting_callback<T>(
    mut writer: RingWriter,
    channels: u16,
) -> impl FnMut(&[T], &cpal::InputCallbackInfo) + Send + 'static
where
    T: cpal::SizedSample,
    f32: FromSample<T>,
{
    let mut scratch = vec![0.0f32; CONVERT_SCRATCH];
    // Convert in whole frames so a chunk boundary never splits a frame.
    let frame = channels.max(1) as usize;
    let stride = (CONVERT_SCRATCH / frame) * frame;

    move |data: &[T], _: &cpal::InputCallbackInfo| {
        for block in data.chunks(stride) {
            let out = &mut scratch[..block.len()];
            for (dst, src) in out.iter_mut().zip(block) {
                *dst = f32::from_sample_(*src);
            }
            writer.write_interleaved(out, channels);
        }
    }
}

impl AudioSource for MicSource {
    fn start(&mut self) -> Result<StreamHandle> {
        if self.stream.is_some() {
            return Err(AudioError::AlreadyRunning);
        }

        let host = cpal::default_host();
        let device = match &self.device_name {
            Some(wanted) => host
                .input_devices()
                .map_err(map_cpal_error)?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == wanted)
                        .unwrap_or(false)
                })
                .ok_or(AudioError::NoDevice("microphone"))?,
            None => host
                .default_input_device()
                .ok_or(AudioError::NoDevice("microphone"))?,
        };

        let supported = device.default_input_config().map_err(map_cpal_error)?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let channels = config.channels;
        let sample_rate = config.sample_rate;

        let info = SourceInfo {
            kind: StreamKind::Mic,
            device_name: device
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| "unknown".into()),
            sample_rate,
            channels,
        };

        let stats = Arc::new(StreamStats::default());
        let (producer, consumer) = rtrb::RingBuffer::new(ring_capacity(
            sample_rate,
            crate::source::DEFAULT_RING_SECONDS,
        ));
        let writer = RingWriter::new(producer, Arc::clone(&stats));

        let on_error = |err: cpal::Error| {
            // An Xrun means the backend itself dropped samples; everything else
            // is a stream we will have to rebuild.
            tracing::warn!(kind = ?err.kind(), "microphone stream error: {err}");
        };

        let stream = match sample_format {
            // The common case on macOS: hand the slice straight to the ring
            // with no conversion and no copy.
            SampleFormat::F32 => {
                let mut writer = writer;
                device.build_input_stream(
                    config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        writer.write_interleaved(data, channels);
                    },
                    on_error,
                    None,
                )
            }
            SampleFormat::I16 => {
                device.build_input_stream(config, converting_callback::<i16>(writer, channels), on_error, None)
            }
            SampleFormat::I32 => {
                device.build_input_stream(config, converting_callback::<i32>(writer, channels), on_error, None)
            }
            other => {
                return Err(AudioError::Unsupported(format!(
                    "microphone sample format {other:?}"
                )));
            }
        }
        .map_err(map_cpal_error)?;

        stream.play().map_err(map_cpal_error)?;
        self.stream = Some(stream);

        Ok(StreamHandle {
            info,
            consumer,
            stats,
        })
    }

    fn stop(&mut self) {
        // Dropping the cpal stream stops it and releases the device.
        self.stream = None;
    }

    fn kind(&self) -> StreamKind {
        StreamKind::Mic
    }
}
