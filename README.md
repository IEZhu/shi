# Shi — local meeting transcriber

Live transcription of any online meeting, entirely on your own machine: who
spoke, when, and what — saved as Markdown.

Microphone and system output are captured as **two independent streams**. The
microphone carries one known speaker; the system output carries every remote
participant already mixed by the conferencing app. Only the second needs
diarization, which removes half the hard problem before a model runs.

## Status

**M0 complete** — capture core and readiness panel.

| Milestone | State |
|---|---|
| M0 — capture + readiness panel | done |
| M1 — transcription (sherpa-onnx, VAD, hybrid partial/final) | next |
| M2 — diarization and voice profiles | |
| M3 — model manager, echo detection, retention | |
| M4 — Windows and Linux ports | |

## Requirements

macOS 14.2 or later (the Core Audio process-tap API), Rust 1.92, Node 22+.

## Running

```bash
npm install
npx tauri build --debug --bundles app
open target/debug/bundle/macos/Shi.app
```

### Why not `tauri dev`

`tauri dev` runs a bare binary with no `Info.plist`. macOS charges the
system-audio capture grant to the *responsible process* — for anything started
from a shell that is the terminal — and when the grant is missing it returns an
endless stream of **zeroes rather than an error**. Capture appears to work and
records silence.

So system audio must be exercised from a real bundle. `tauri dev` is still fine
for frontend work, where the microphone alone is enough.

For the lower-level capture examples there is a wrapper that bundles, ad-hoc
signs and launches in one step:

```bash
scripts/dev-run.sh readiness 15
```

The same trap is why `StreamStats::has_signal` exists and why the readiness
panel reports **silent** as a distinct verdict from **running**. Measurements
behind all of this are in [docs/capture-macos.md](docs/capture-macos.md).

## Testing

```bash
cargo test --workspace
```

Every test runs with **no audio hardware**, replaying WAV fixtures through
`FileSource`. That is a deliberate invariant: if a test above the capture layer
ever needs a real device, a platform assumption has leaked into code that must
stay portable — which is exactly what would turn the Windows and Linux ports
into a rewrite instead of one new `AudioSource`.

## Layout

```
crates/audio/        AudioSource trait, ring buffers, mic (cpal), WAV FileSource
crates/audio/objc/   Core Audio process-tap shim (macOS)
src-tauri/           app shell, capture orchestration, readiness events
ui/                  React frontend
scripts/dev-run.sh   bundle + sign + launch, for testing system capture
docs/                findings worth keeping
```
