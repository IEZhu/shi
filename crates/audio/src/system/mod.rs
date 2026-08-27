//! System-output capture — the one part of capture that is genuinely
//! per-platform.
//!
//! Everything else in this crate is shared. Porting means adding a module
//! here and wiring it into `SystemSource` below; nothing downstream changes.

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::MacOsTapSource as SystemSource;

#[cfg(not(target_os = "macos"))]
mod unsupported;

#[cfg(not(target_os = "macos"))]
pub use unsupported::UnsupportedSystemSource as SystemSource;
