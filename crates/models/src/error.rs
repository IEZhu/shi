use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("unknown model: {0}")]
    Unknown(String),

    #[error("downloading {url}: {source}")]
    Transport {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },

    #[error("downloading {url}: server replied {status}")]
    Status { url: String, status: u16 },

    #[error(
        "{name} does not match its expected checksum — the download is corrupt or the file \
         upstream has changed. Nothing was installed."
    )]
    ChecksumMismatch { name: String },

    #[error("unpacking {}: {source}", path.display())]
    Unpack {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cancelled")]
    Cancelled,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ModelError>;
