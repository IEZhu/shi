use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{ModelError, Result};
use crate::registry::{Install, ModelSpec};

/// Read size. Large enough that a 500 MB download is not a syscall benchmark.
const CHUNK: usize = 256 * 1024;

/// How far along a download is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub downloaded: u64,
    pub total: u64,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.downloaded as f32 / self.total as f32).clamp(0.0, 1.0)
    }
}

/// Fetch and install a model, resuming an interrupted attempt if one is there.
///
/// Nothing lands in the models directory until the bytes have been checked, so
/// an interrupted or corrupted download can never present itself as a working
/// model — which would otherwise surface much later as an unexplained failure
/// to start a meeting.
///
/// `should_continue` is polled between chunks so the UI can cancel; returning
/// false leaves the partial file in place for the next attempt.
pub fn install(
    spec: &ModelSpec,
    models_dir: &Path,
    mut on_progress: impl FnMut(Progress),
    mut should_continue: impl FnMut() -> bool,
) -> Result<PathBuf> {
    fs::create_dir_all(models_dir)?;

    let partial = models_dir.join(format!(".{}.part", spec.id));
    let already = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

    // Resuming saves the user from starting a 500 MB download over because a
    // hotel network dropped once.
    let mut request = ureq::get(spec.url);
    if already > 0 {
        request = request.header("Range", &format!("bytes={already}-"));
    }

    let mut response = request.call().map_err(|source| ModelError::Transport {
        url: spec.url.to_string(),
        source: Box::new(source),
    })?;

    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(ModelError::Status {
            url: spec.url.to_string(),
            status,
        });
    }

    // A server that ignores Range answers 200 with the whole file, in which
    // case the partial bytes are worthless and must not be prepended.
    let resuming = already > 0 && status == 206;
    let mut file = if resuming {
        let mut file = fs::OpenOptions::new().write(true).open(&partial)?;
        file.seek(SeekFrom::End(0))?;
        file
    } else {
        File::create(&partial)?
    };

    let mut downloaded = if resuming { already } else { 0 };
    on_progress(Progress {
        downloaded,
        total: spec.download_bytes,
    });

    let mut reader = response
        .body_mut()
        .with_config()
        // The default cap is far below a model; this is a known size we chose.
        .limit(spec.download_bytes.saturating_add(CHUNK as u64))
        .reader();

    let mut buffer = vec![0u8; CHUNK];
    loop {
        if !should_continue() {
            file.flush()?;
            return Err(ModelError::Cancelled);
        }

        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        on_progress(Progress {
            downloaded,
            total: spec.download_bytes,
        });
    }
    file.flush()?;
    drop(file);

    // Hash the finished file rather than the stream: a resumed download has
    // bytes we never saw this time round.
    let actual = sha256_of(&partial)?;
    if actual != spec.integrity.expected() {
        // Keep nothing: a wrong file that survives will be resumed forever.
        let _ = fs::remove_file(&partial);
        return Err(ModelError::ChecksumMismatch {
            name: spec.display_name.to_string(),
        });
    }

    let destination = spec.path_in(models_dir);
    match spec.install {
        Install::File(_) => {
            fs::rename(&partial, &destination)?;
        }
        Install::Archive(_) => {
            unpack(&partial, models_dir)?;
            let _ = fs::remove_file(&partial);
        }
    }

    tracing::info!(
        model = spec.id,
        verified = if spec.integrity.is_published() { "published digest" } else { "recorded digest" },
        "model installed"
    );
    Ok(destination)
}

/// Remove an installed model.
pub fn uninstall(spec: &ModelSpec, models_dir: &Path) -> Result<()> {
    let path = spec.path_in(models_dir);
    match spec.install {
        Install::File(_) if path.is_file() => fs::remove_file(&path)?,
        Install::Archive(_) if path.is_dir() => fs::remove_dir_all(&path)?,
        _ => {}
    }
    Ok(())
}

fn sha256_of(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unpack(archive: &Path, into: &Path) -> Result<()> {
    let file = File::open(archive)?;
    let decoder = bzip2::read::BzDecoder::new(file);
    tar::Archive::new(decoder)
        .unpack(into)
        .map_err(|source| ModelError::Unpack {
            path: archive.to_path_buf(),
            source,
        })
}
