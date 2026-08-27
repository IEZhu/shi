//! Download and verification.
//!
//! The network test is opt-in (`SHI_NETWORK_TESTS=1`) so an offline build is
//! not a failing build; everything that can be checked without a server is
//! checked unconditionally.

use std::path::Path;

use shi_models::{ModelError, by_id, install, uninstall};

fn never_cancel() -> bool {
    true
}

#[test]
fn an_absent_model_is_reported_as_not_installed() {
    let dir = tempfile::tempdir().expect("tempdir");
    for spec in shi_models::CATALOGUE {
        assert!(
            !spec.installed(dir.path()),
            "{} claimed to be installed in an empty directory",
            spec.id
        );
    }
}

#[test]
fn an_empty_directory_left_by_a_failed_unpack_is_not_an_installation() {
    // The failure this guards against: a half-finished extraction that looks
    // installed, so the app skips the download and then cannot start a meeting.
    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("parakeet-tdt-0.6b-v3-int8").expect("in catalogue");
    std::fs::create_dir_all(spec.path_in(dir.path())).expect("mkdir");

    assert!(
        !spec.installed(dir.path()),
        "an empty model directory must not count as installed"
    );
}

#[test]
fn uninstalling_something_absent_is_not_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    for spec in shi_models::CATALOGUE {
        uninstall(spec, dir.path()).expect("uninstall should be idempotent");
    }
}

#[test]
fn cancelling_leaves_nothing_installed() {
    if std::env::var("SHI_NETWORK_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping: set SHI_NETWORK_TESTS=1 to run network tests");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("silero-vad").expect("in catalogue");

    let mut ticks = 0;
    let result = install(spec, dir.path(), |_| {}, || {
        ticks += 1;
        ticks < 2
    });

    assert!(matches!(result, Err(ModelError::Cancelled)), "{result:?}");
    assert!(
        !spec.installed(dir.path()),
        "a cancelled download must not present itself as a working model"
    );
}

#[test]
fn a_verified_download_installs_and_reports_progress() {
    if std::env::var("SHI_NETWORK_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping: set SHI_NETWORK_TESTS=1 to run network tests");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("silero-vad").expect("in catalogue");

    let mut seen = Vec::new();
    let path = install(spec, dir.path(), |p| seen.push(p), never_cancel).expect("install");

    assert!(path.is_file(), "the model should be where it says it is");
    assert!(spec.installed(dir.path()));
    assert!(seen.len() >= 2, "progress should be reported as it goes");

    let last = seen.last().expect("progress");
    assert_eq!(last.downloaded, spec.download_bytes);
    assert!((last.fraction() - 1.0).abs() < 1e-6);

    // And removing it puts things back.
    uninstall(spec, dir.path()).expect("uninstall");
    assert!(!spec.installed(dir.path()));
    assert!(!Path::new(&path).exists());
}
