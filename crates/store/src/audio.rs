use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use shi_audio::StreamKind;

use crate::error::{Result, StoreError};

/// Everything downstream works at this rate, and so does what we keep.
pub const RECORDING_SAMPLE_RATE: u32 = 16_000;

/// Extension used while a meeting is in progress.
///
/// Headerless raw PCM, not WAV, and deliberately so. A WAV header records the
/// data length, which is written when the file is closed — so a recording
/// interrupted by a crash claims to hold zero samples, and any recovery that
/// trusts the header throws the audio away. Raw PCM has nothing to be wrong:
/// a truncated file is simply a shorter recording.
const IN_PROGRESS: &str = "pcm";
/// Extension once the meeting is closed and the audio compressed.
const COMPRESSED: &str = "flac";

/// Where meeting audio is kept.
///
/// Audio exists so a meeting can be re-diarized or re-transcribed later with a
/// better model. It is written uncompressed while the meeting runs — streaming
/// FLAC would mean holding a whole meeting in memory — and compressed when the
/// meeting closes. A recording interrupted by a crash is left as WAV and
/// compressed on the next launch rather than lost.
pub struct AudioStore {
    root: PathBuf,
}

impl AudioStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn meeting_dir(&self, meeting_id: i64) -> PathBuf {
        self.root.join(meeting_id.to_string())
    }

    /// Begin recording one stream of a meeting.
    pub fn recorder(&self, meeting_id: i64, stream: StreamKind) -> Result<Recorder> {
        let dir = self.meeting_dir(meeting_id);
        std::fs::create_dir_all(&dir).map_err(|source| StoreError::Write {
            path: dir.display().to_string(),
            source,
        })?;
        Recorder::create(dir.join(format!("{}.{IN_PROGRESS}", stream.as_str())))
    }

    /// Compress anything an interrupted run left behind.
    ///
    /// Returns how many files were converted. Called at startup, because the
    /// alternative is a meetings folder that silently grows at seven times the
    /// intended rate.
    pub fn compress_orphans(&self) -> usize {
        let mut converted = 0;
        let Ok(meetings) = std::fs::read_dir(&self.root) else {
            return 0;
        };

        for meeting in meetings.flatten() {
            let Ok(files) = std::fs::read_dir(meeting.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().is_none_or(|e| e != IN_PROGRESS) {
                    continue;
                }
                match compress(&path) {
                    Ok(_) => converted += 1,
                    Err(err) => tracing::warn!("cannot compress {}: {err}", path.display()),
                }
            }
        }

        if converted > 0 {
            tracing::info!(converted, "compressed audio left by an interrupted run");
        }
        converted
    }

    /// Delete the audio of the given meetings, leaving their transcripts alone.
    ///
    /// Transcripts are small and are the point of the app; audio is a means to
    /// re-processing and is what actually fills a disk.
    pub fn prune(&self, meeting_ids: &[i64]) -> u64 {
        let mut freed = 0;
        for id in meeting_ids {
            let dir = self.meeting_dir(*id);
            if !dir.is_dir() {
                continue;
            }
            freed += directory_size(&dir);
            if let Err(err) = std::fs::remove_dir_all(&dir) {
                tracing::warn!("cannot remove {}: {err}", dir.display());
            }
        }
        if freed > 0 {
            tracing::info!(
                meetings = meeting_ids.len(),
                freed_bytes = freed,
                "pruned meeting audio"
            );
        }
        freed
    }

    /// How much disk the stored audio occupies.
    pub fn usage_bytes(&self) -> u64 {
        directory_size(&self.root)
    }

    /// The audio files kept for a meeting, if any.
    pub fn files_for(&self, meeting_id: i64) -> Vec<PathBuf> {
        let dir = self.meeting_dir(meeting_id);
        std::fs::read_dir(dir)
            .map(|entries| entries.flatten().map(|e| e.path()).collect())
            .unwrap_or_default()
    }
}

/// Writes 16 kHz mono audio for one stream of one meeting.
///
/// Samples go out as raw little-endian i16 — the format is fixed and known, so
/// nothing about the file needs to be corrected when it is closed. That is what
/// makes an interrupted recording recoverable instead of merely present.
pub struct Recorder {
    writer: Option<BufWriter<File>>,
    path: PathBuf,
    failed: bool,
}

impl Recorder {
    fn create(path: PathBuf) -> Result<Self> {
        let file = File::create(&path).map_err(|source| StoreError::Write {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self {
            writer: Some(BufWriter::new(file)),
            path,
            failed: false,
        })
    }

    /// Append samples. Called from the pipeline worker, never the audio thread.
    pub fn write(&mut self, samples: &[f32]) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };

        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for sample in samples {
            let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&scaled.to_le_bytes());
        }

        if writer.write_all(&bytes).is_err() {
            // Report once: a full disk would otherwise produce one line per
            // block for the rest of the meeting.
            if !self.failed {
                tracing::error!("recording to {} failed", self.path.display());
                self.failed = true;
            }
            self.writer = None;
        }
    }

    /// Close the file and compress it.
    pub fn finish(mut self) -> Result<PathBuf> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush().map_err(|source| StoreError::Write {
                path: self.path.display().to_string(),
                source,
            })?;
        }
        compress(&self.path)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Whatever reached the file is a valid recording; the startup sweep
        // compresses it. Nothing here needs to succeed for that to hold.
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.flush();
        }
    }
}

/// Compress a finished WAV to FLAC and remove the original.
///
/// Lossless on purpose: the whole reason to keep audio is to re-run models over
/// it, and a lossy codec would mean a later transcript was made from something
/// slightly different than the meeting.
fn compress(pcm: &Path) -> Result<PathBuf> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let write_error = |path: &Path, err: std::io::Error| StoreError::Write {
        path: path.display().to_string(),
        source: err,
    };

    let raw = std::fs::read(pcm).map_err(|err| write_error(pcm, err))?;
    // A crash can truncate mid-sample; drop the stray byte rather than refuse.
    let samples: Vec<i32> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
        .collect();

    let flac = pcm.with_extension(COMPRESSED);

    // Nothing was ever recorded; there is no audio to lose.
    if samples.is_empty() {
        let _ = std::fs::remove_file(pcm);
        return Ok(flac);
    }

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|err| write_error(pcm, std::io::Error::other(format!("{err:?}"))))?;
    let source = flacenc::source::MemSource::from_samples(
        &samples,
        1,
        16,
        RECORDING_SAMPLE_RATE as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| write_error(pcm, std::io::Error::other(format!("{err:?}"))))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|err| write_error(&flac, std::io::Error::other(format!("{err:?}"))))?;
    std::fs::write(&flac, sink.as_slice()).map_err(|err| write_error(&flac, err))?;

    // Only now is the original expendable.
    std::fs::remove_file(pcm).map_err(|err| write_error(pcm, err))?;
    Ok(flac)
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => directory_size(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}
