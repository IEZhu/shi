# System audio capture on macOS — what the experiments showed

Findings verified on macOS 26.3 (arm64) while building M0. Everything here was
measured, not recalled.

## The API is 14.2, not 14.4

`AudioHardwareTapping.h` in the installed SDK annotates both
`AudioHardwareCreateProcessTap` and `AudioHardwareDestroyProcessTap` with
`API_AVAILABLE(macos(14.2))`. The deployment target is set accordingly in
`crates/audio/build.rs`.

## The tap headers are not in the umbrella header

`<CoreAudio/CoreAudio.h>` pulls in only `AudioHardware.h` and `HostTime.h`. The
tap API lives in ObjC-only headers that must be imported explicitly:

```objc
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
```

## A refused capture returns silence, not an error

This is the single most important finding, and it shaped the design.

Run the same binary two ways:

| Launched as | `CreateProcessTap` | `CreateAggregate` | IOProc runs | Samples |
|---|---|---|---|---|
| CLI from a terminal | `noErr` | `noErr` | yes, 48 kHz | **all zero** |
| Signed `.app` via `open` | `noErr` | `noErr` | yes, 48 kHz | real audio |

Every status code says success in both cases. Frame counts are identical and
correct. Only the sample values differ.

The cause is TCC attribution: the audio-capture grant
(`kTCCServiceAudioCapture`, gated by the `NSAudioCaptureUsageDescription`
Info.plist key — both strings confirmed inside `tccd` itself) is charged to the
*responsible process*. For a shell-launched binary that is the terminal, which
does not hold the grant, so the tap yields an endless stream of zeroes.

Two consequences, both load-bearing:

1. **"Stream started" proves nothing.** The readiness panel must assert that a
   non-zero sample has actually been seen. `StreamStats::has_signal` exists for
   exactly this, and `silence_never_sets_the_signal_flag` pins the behaviour.
2. **`cargo run` cannot test system capture.** Use `scripts/dev-run.sh <example>`,
   which bundles, ad-hoc signs, and launches as an app.

## Aggregate device shape

A mono global tap produces exactly what the pipeline wants, with no reshaping:

```
tap ASBD:                48000 Hz, 1 channel, 32-bit float, packed
aggregate input config:  1 buffer, 1 channel, 2048 bytes (512 frames)
callback rate:           ~93/s  (512 frames @ 48 kHz)
```

Measured against a generated 10 s / 0.5-amplitude 440 Hz tone, every callback
reported a peak of 0.49997 — the tap is sample-accurate, not approximate.

Note `kAudioSubTapDriftCompensationKey` *does* exist (`"drift"`), despite being
easy to miss; the aggregate sets it so the tap does not drift against the
hardware clock.

## Streams really are independent

Playing one tone through the speakers while both streams ran:

- system stream: flat 0.5 — the digital tap of the tone
- microphone:    fluctuating 0.05–0.2 — acoustic pickup plus room noise

Different signals from one sound. This independence is what lets the pipeline
skip diarization on the microphone entirely.

It also shows the echo problem in miniature: without headphones the microphone
*does* pick up the remote side, so the same utterance can be transcribed twice.
`cpal`'s `DeviceDescription::interface_type()` reports `InterfaceType::BuiltIn`
vs `Bluetooth`/`Usb`, which is the cheap first half of the echo warning.

## The tap goes quiet, and the clock has to know

The system tap delivers nothing while no application is playing. Counting only
the samples that arrive therefore makes the pipeline's idea of "now" short by
the whole idle period.

Measured on a meeting left silent for fifteen seconds before anyone spoke:

| | sample clock only | reconciled against the wall clock |
|---|---:|---:|
| first utterance stamped at | 422 ms | 9 510 ms |
| microphone audio recorded | 37.5 s | 37.1 s |
| system audio recorded | 28.0 s | 35.0 s |

Two consequences, both real:

- every remote line was stamped early by however long the room had been quiet,
  so wall-clock timestamps drifted further from the calendar as a meeting went on
- the two streams ran on different timelines, so microphone and system
  utterances interleaved in the wrong order in the transcript

`StreamPipeline` now compares elapsed time against the audio it has actually
heard and inserts silence when it falls behind, feeding it to the voice-activity
detector and the recording alike so the transcript and the audio agree about
when things happened. It only ever adds: a fixture replayed faster than real
time is left alone, which is what keeps the tests deterministic.
