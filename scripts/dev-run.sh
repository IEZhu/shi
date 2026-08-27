#!/usr/bin/env bash
# Build a Rust example into a signed .app bundle and launch it as an app.
#
# System-audio capture cannot be tested with `cargo run`. macOS attributes the
# capture request to the *responsible process*, which for a binary started from
# a shell is the terminal — and a terminal without the grant gets an endless
# stream of zeroes rather than an error. Only a real bundle launched as an app
# is attributed to itself.
#
#   scripts/dev-run.sh readiness [args...]
set -euo pipefail

EXAMPLE="${1:?usage: dev-run.sh <example-name> [args...]}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_ID="io.github.iezhu.shi.dev.${EXAMPLE}"
APP="${ROOT}/target/dev-bundles/${EXAMPLE}.app"
LOG="${ROOT}/target/dev-bundles/${EXAMPLE}.log"

cargo build --manifest-path "${ROOT}/Cargo.toml" -p shi-audio --example "${EXAMPLE}"
BIN="$(find "${ROOT}/target/debug/examples" -maxdepth 1 -name "${EXAMPLE}-*" -type f -perm +111 \
       -exec ls -t {} + | head -1)"

rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS"
cp "${BIN}" "${APP}/Contents/MacOS/${EXAMPLE}"

cat > "${APP}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
    <key>CFBundleName</key><string>${EXAMPLE}</string>
    <key>CFBundleExecutable</key><string>${EXAMPLE}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSMinimumSystemVersion</key><string>14.2</string>
    <key>LSBackgroundOnly</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Transcribes what you say during meetings, locally on this Mac.</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>Transcribes what other participants say during meetings, locally on this Mac.</string>
</dict>
</plist>
PLIST

codesign --force --sign - --identifier "${BUNDLE_ID}" "${APP}" >/dev/null 2>&1

: > "${LOG}"
open -a "${APP}" --env "DEV_LOG=${LOG}" --args "$@"
echo "launched ${APP}"
echo "log: ${LOG}"
