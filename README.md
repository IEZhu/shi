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
| M1 — transcription, storage, live transcript, echo suppression | done |
| M2 — diarization and persistent voice profiles | done |
| M3 — model manager, retention, hotword glossary | next |
| M4 — Windows and Linux ports | |

Meetings transcribe end to end and are written to Markdown as they happen.
Remote participants are separated by voice; name one and the whole transcript
updates, and the same person is recognised automatically in later meetings.

## Requirements

macOS 14.2 or later (the Core Audio process-tap API), Rust 1.92, Node 22+.

## Running

```bash
npm install
scripts/fetch-models.sh
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
crates/pipeline/     resampling, VAD, recognition, draft cadence, echo suppression,
                     speaker tracking
crates/store/        SQLite schema and the Markdown projection
src-tauri/           app shell, session orchestration, commands
ui/                  React frontend
scripts/dev-run.sh   bundle + sign + launch, for testing system capture
scripts/fetch-models.sh  download the recogniser and VAD
docs/                findings worth keeping
```
