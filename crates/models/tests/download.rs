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
fn a_half_extracted_model_is_not_an_installation() {
    // This is the failure that crashed the app: an extraction interrupted
    // partway leaves every expected file, with the right names, in the right
    // place — and a truncated one among them. Judging by the files passes, and
    // handing a truncated model to the runtime aborts the process instead of
    // returning an error. Only a receipt written after everything finished can
    // tell the two apart.
    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("parakeet-tdt-0.6b-v3-int8").expect("in catalogue");

    let model = spec.path_in(dir.path());
    std::fs::create_dir_all(&model).expect("mkdir");
    for name in [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        // Plausible but incomplete, exactly as tar leaves them mid-extraction.
        std::fs::write(model.join(name), vec![0u8; 4096]).expect("write");
    }

    assert!(
        !spec.installed(dir.path()),
        "a partly extracted model claimed to be installed"
    );

    // An empty directory must not count either.
    let empty = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(spec.path_in(empty.path())).expect("mkdir");
    assert!(!spec.installed(empty.path()));
}

#[test]
fn a_receipt_without_the_model_is_not_an_installation() {
    // The mirror image: a receipt left behind by a removal that half finished.
    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("silero-vad").expect("in catalogue");
    std::fs::write(spec.receipt_in(dir.path()), "whatever").expect("write receipt");
    assert!(!spec.installed(dir.path()));
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
    assert!(
        spec.receipt_in(dir.path()).is_file(),
        "a finished install must leave a receipt"
    );
    assert!(seen.len() >= 2, "progress should be reported as it goes");

    let last = seen.last().expect("progress");
    assert_eq!(last.downloaded, spec.download_bytes);
    assert!((last.fraction() - 1.0).abs() < 1e-6);

    // And removing it puts things back, receipt included.
    uninstall(spec, dir.path()).expect("uninstall");
    assert!(!spec.installed(dir.path()));
    assert!(!Path::new(&path).exists());
    assert!(!spec.receipt_in(dir.path()).exists());
}

#[test]
fn a_download_that_completed_but_never_installed_is_not_restarted() {
    // A previous attempt can leave a complete .part behind by failing during
    // verification or extraction — which is exactly what happened when an app
    // was killed mid-extraction. Asking the server to resume from the end of a
    // complete file earns a 416, so that state has to be recognised instead.
    if std::env::var("SHI_NETWORK_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping: set SHI_NETWORK_TESTS=1 to run network tests");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let spec = by_id("silero-vad").expect("in catalogue");

    // Fetch it once, then reconstruct the state a crashed install leaves.
    let path = install(spec, dir.path(), |_| {}, never_cancel).expect("first install");
    let bytes = std::fs::read(&path).expect("read");
    uninstall(spec, dir.path()).expect("uninstall");

    let partial = dir.path().join(format!(".{}.part", spec.id));
    std::fs::write(&partial, &bytes).expect("stage a complete part");

    // This must succeed from the file on disk rather than asking to resume.
    install(spec, dir.path(), |_| {}, never_cancel).expect("install from a complete part");
    assert!(spec.installed(dir.path()));
    assert!(!partial.exists(), "the staged download should be consumed");
}
