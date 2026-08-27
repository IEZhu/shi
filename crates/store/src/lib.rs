//! Meeting persistence.
//!
//! SQLite holds the truth and Markdown is rendered from it, never appended to.
//! That ordering is deliberate: identifying a speaker after the meeting has to
//! fix every line they spoke, which is one UPDATE and a re-render rather than a
//! rewrite of a file the user may have already edited.

pub mod error;
pub mod markdown;
pub mod model;
mod schema;
pub mod store;

pub use error::{Result, StoreError};
pub use markdown::{MarkdownOptions, render, write_to};
pub use model::{Meeting, NewSegment, Segment};
pub use store::Store;
