//! Getting the models onto the machine.
//!
//! Downloads are resumable and always verified before anything is installed,
//! because a truncated model does not fail loudly — it fails when the user
//! tries to start a meeting.

pub mod download;
pub mod error;
pub mod registry;

pub use download::{Progress, install, uninstall};
pub use error::{ModelError, Result};
pub use registry::{
    CATALOGUE, Family, Install, Integrity, ModelKind, ModelSpec, by_id, required,
};
