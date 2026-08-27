fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    println!("cargo:rerun-if-changed=objc/shi_tap.m");
    println!("cargo:rerun-if-changed=objc/shi_tap.h");

    cc::Build::new()
        .file("objc/shi_tap.m")
        .flag("-fobjc-arc")
        // The tap API landed in macOS 14.2; the readiness check reports a
        // clear message on older systems rather than failing to link.
        .flag("-mmacosx-version-min=14.2")
        .warnings(true)
        .compile("shi_tap");

    for framework in ["CoreAudio", "AudioToolbox", "Foundation"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}
