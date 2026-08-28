# Speaker echo — what the measurements showed

Without headphones the microphone re-records whatever the speakers play, so
every remote utterance reaches the transcript twice: once from the digital tap,
once, worse, through the air. `EchoReference` compares the loudness contour of
each microphone utterance against the system stream around the same moment and
discards the ones that match.

It passed its unit tests from the day it was written and never once fired in a
real meeting. Everything below was measured with `scripts/dev-run.sh echo_probe`,
which plays speech through the speakers with nobody talking — so every
microphone utterance is, by definition, an echo.

## What the first real run said

| | |
|---|---:|
| system utterances | 7 |
| microphone utterances reaching the transcript | 7 |
| suppressed | **0** |

Every line duplicated. The scores explain why:

```
0.045  0.078  0.081  0.232  0.257  0.287  0.374  0.404  0.478  0.504  0.505
```

against a threshold of 0.7. Nothing came close. But the timestamps are the
real tell:

| system said it at | microphone heard it at | apart |
|---:|---:|---:|
| 7.08 s | 8.04 s | 0.96 s |
| 9.48 s | 10.41 s | 0.93 s |
| 12.01 s | 12.90 s | 0.90 s |
| 14.37 s | 15.33 s | 0.96 s |

The two streams were running about 0.93 s apart, and the detector is only
allowed to search ±400 ms. It was never comparing the right two moments.

## Two causes, both about time rather than audio

**The reference skipped its own silence.** When a source delivers nothing, the
pipeline inserts silence so its clock stays honest — into the voice-activity
detector and into the recording, but not into the echo reference. The reference
is indexed by how much audio it has been handed, so every quiet stretch slid it
earlier by exactly that much. With the system tap only running while something
played (see [capture-macos.md](capture-macos.md)), that was most of a meeting.

**Each stream carried a backlog.** A source starts filling its ring the moment
it is opened; the pipeline's clock started later, after a model load taking
seconds. Everything captured in between was stamped as though it happened at
time zero, pushing that stream's whole timeline forward by the size of its
backlog — and the two backlogs differed, because the two sources do not open at
the same instant:

```
discarded backlog: mic 67584 samples (1.41 s), system 3072 samples (0.06 s)
```

1.34 s of difference, from nothing but the order the two devices happened to
open in. The recogniser now loads *before* the microphone opens, so nothing
said after pressing record is lost to it, and whatever little is left in the
rings is dropped before the clock starts.

## What it says now

Same probe, same room, same speakers:

| | |
|---|---:|
| system utterances | 7 |
| microphone utterances reaching the transcript | **0** |
| suppressed | 7 |

Real echo scores, aligned: **0.816 – 0.942**. Unrelated audio, from the
misaligned run above: **at most 0.505**. The 0.7 threshold sits in the middle of
a gap of roughly 0.3, which is where a threshold should sit — and this is now
measured on a real loudspeaker and a real microphone rather than on the
simulated room the unit tests use.

## The case that decides whether this is safe to ship

Suppression deletes speech. The dangerous failure is not a missed echo but a
silenced user, so double talk — the microphone carrying the user's own voice
*and* the speakers at once — is pinned by
`speaking_over_the_echo_is_not_suppressed`. The user survives.

Every echo decision is logged with the score that made it. If suppression ever
misfires, the number that decided it is in the log rather than left to guesswork.
