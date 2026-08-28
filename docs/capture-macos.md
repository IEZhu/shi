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

What it took to make suppression actually fire on a real loudspeaker is in
[echo.md](echo.md) — and it turned out to be about clocks, not about audio.

## An aggregate with only a tap in it has no clock

This is the defect behind "поток запущен, но устройство не прислало ни одного
кадра", and it hid for a long time because every measurement that looked fine
had something playing at the time.

The aggregate was built with an empty `kAudioAggregateDeviceSubDeviceListKey`
and the default output named only in `kAudioAggregateDeviceMainSubDeviceKey` —
a device that is the clock master but not a member. Such an aggregate has no
hardware to run on, so its IO proc is driven by whoever happens to be playing:

| system state | frames in 20 s |
|---|---:|
| nothing playing | **0** |
| playback starts at t=5 s | 6.3 s of frames in a 12.1 s window |

Every status code says success throughout. `AudioDeviceStart` returns `noErr`,
the tap exists, the aggregate exists — and nothing ever arrives.

Putting a real device *inside* the sub-device list fixes it. Verified with
`AudioObjectGetPropertyData` on the created aggregate:

```
requested sub-device: BuiltInSpeakerDevice
full sub-device list: (BuiltInSpeakerDevice)
active sub-devices:   1 -> BuiltInSpeakerDevice
aggregate input:      1 buffer(s): [0]=1ch      <- the tap
aggregate output:     1 buffer(s): [0]=2ch      <- the device
```

Frames then arrive continuously: five consecutive runs on a silent machine gave
6.0 s of frames per 6 s run, and one uninterrupted run held ~94 IO callbacks per
second for 100 s without a gap.

### Which device to clock from

Not the current default output: AirPods disconnect and docks get unplugged.
The tap is *global* — it follows processes, not one device — so the clock only
has to be stable, not the one the user is listening through. Measured by
switching the default output to another device for 9 s in the middle of a 24 s
capture: 24.0 s of frames, meter pegged throughout, not a frame lost.

So the shim prefers a built-in output, which cannot be unplugged, then a
built-in input, then the current output. Output first on purpose: an input
sub-device would tie system-audio capture to microphone access, and measurement
says it buys nothing — a built-in output clocks the aggregate just as steadily.

### The clock device's inputs come first

An aggregate presents its sub-device's input buffers ahead of the tap's, so
`mBuffers[0]` is only the tap when the clock device has no inputs of its own.
The shim asks the clock device how many input buffers it contributes and indexes
past them.

That the resulting stream is really the tap — and not, say, the microphone
sitting next to it in the aggregate — was checked against a generated
440 Hz tone at amplitude 0.25 played through the speakers:

```
peak=0.2500   rms=0.17674   dominant: 440 Hz (100.0%), harmonics 0.0%
```

Bit-exact amplitude and a theoretical RMS of 0.17678. A microphone would have
delivered room noise and speaker harmonics at an attenuated level.

The offset itself was exercised by clocking from the built-in *microphone*
instead, which contributes one input buffer and puts the tap at index 1: same
tone, same bit-exact result. Built-in speakers contribute none, so on this Mac
the shipping path indexes 0.

### A tap can start clean and still never run

Seen twice while measuring the above: every status `noErr`, the sub-device
listed as active, `AudioDeviceStart` returning `noErr` — and the IO proc never
called once. It did not reproduce in the signed-app configuration (5 of 5 runs
healthy), but once frames do start they never stop, so the failure lives
entirely in startup. `MacOsTapSource::start` therefore waits up to 700 ms for a
first frame and rebuilds the tap if none comes, up to three times. On the last
attempt it keeps the stream regardless: a tap that has not delivered yet may
still wake on playback, and the readiness panel reports `NoFrames` either way.

## The tap used to go quiet, and the clock had to know

Before the aggregate had a clock of its own, the tap delivered nothing while no
application was playing. Counting only the samples that arrive therefore made
the pipeline's idea of "now" short by the whole idle period.

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

`StreamPipeline` compares elapsed time against the audio it has actually heard
and inserts silence when it falls behind, feeding it to the voice-activity
detector and the recording alike so the transcript and the audio agree about
when things happened. It only ever adds: a fixture replayed faster than real
time is left alone, which is what keeps the tests deterministic.

With the aggregate clocked properly the gap no longer opens in the first place.
The reconciliation stays: it costs nothing when the streams are healthy, and it
is the only thing standing between a stalled source and a transcript whose
timestamps quietly stop matching the meeting.
