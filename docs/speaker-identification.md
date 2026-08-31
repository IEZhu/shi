# Choosing a voice model, and the thresholds it needs

The plan named CAM++ from 3D-Speaker as the embedding model and refused to
commit to a cosine threshold, on the grounds that the right value depends on the
model, the microphones and the room. Measuring settled both questions — and
overturned the first.

## The measurement

`cargo run -p shi-pipeline --release --example calibrate -- <corpus> <model>...`

A corpus of clips named `<speaker>_<n>.wav`. Six voices, four different
sentences each, so that similarity within a voice is not inflated by identical
text. Two of the six are deliberately the hard case: same gender, closely
related languages.

## CAM++ was the wrong choice

| Model | dim | same voice, min | other voice, max | margin | EER |
|---|---:|---:|---:|---:|---:|
| wespeaker CAM++ LM | 512 | 0.478 | 0.852 | **−0.374** | 19.4% |
| 3dspeaker CAM++ VoxCeleb | 512 | 0.531 | 0.870 | **−0.339** | 16.7% |
| 3dspeaker CAM++ zh_en advanced | 192 | 0.705 | 0.659 | +0.046 | 0.0% |
| **nemo TitaNet small** | 192 | 0.814 | 0.662 | **+0.151** | 0.0% |

A negative margin means the worst same-voice pair scores *below* the best
different-voice pair: no threshold separates them at all. Both VoxCeleb-trained
CAM++ variants are in that state on this corpus, so the model the plan named
would have produced an identification feature that cannot work.

**TitaNet small** wins on every axis measured — widest margin, smallest model of
the two that work, and fastest.

## The margin is real, not an artefact

Different speakers in the corpus also speak different languages, which could
have separated them for the wrong reason. It did not. The hardest pair for both
surviving models is the one designed to be hard:

```
0.662  Lesya vs Milena     (both female, Ukrainian and Russian)
0.526  Anna vs Lesya
0.451  Anna vs Milena
```

Separation is tightest exactly where the voices are closest, which is what a
speaker model should do. The binding constraint for TitaNet is therefore:

```
hardest different-voice pair   0.662
weakest same-voice match       0.814   (Milena)
                               -----
usable gap                     0.152
```

## Thresholds

The two errors are not symmetric:

- **Attaching the wrong name** is a confident lie written into a document the
  user will trust later.
- **Failing to recognise someone** produces another "Speaker 3" to name again —
  irritating, and self-correcting.

So the thresholds sit high, deliberately biased toward the recoverable error:

| Threshold | Value | Purpose |
|---|---:|---|
| `known` | 0.75 | match a stored voice profile and write a real name |
| `session` | 0.70 | group utterances within one meeting under one slot |

## What this does not prove

The corpus is synthesised speech. Real voices vary more between sessions — a
different microphone, a cold, a different room — so same-voice similarity will
fall, and real speakers can resemble each other more than these do. Expect the
gap to narrow.

That is why the numbers are settings rather than constants, and why the harness
exists: point `calibrate` at a directory of real meeting clips and it re-derives
them. Until that is done, treat 0.75 as a starting point chosen to fail in the
safe direction.

## The first real meeting, and how far off the numbers were

A 71-minute conference call, captured with `meeting_capture` and clustered with
`diarize_sweep`: 184 utterances on the system stream, through a Bluetooth
headset at 16 kHz — the harsh end of the range, not the easy one.

```
pairwise cosine: min -0.165  p10 0.086  median 0.330  p90 0.809  max 0.976
```

The structure predicted above is there: different voices sit near 0.1, the same
voice near 0.8. What the synthetic corpus got wrong is the *height* of that
upper mode. Real same-voice pairs top out around 0.81, so a threshold of 0.70
merges almost nothing:

| threshold | clusters |
|---:|---:|
| 0.40 | 13 |
| 0.45 | 14 |
| 0.50 | 16 |
| 0.60 | 27 |
| **0.70** | **52** |
| 0.80 | 86 |

Ground truth, from a participant: a large call in which two or three people did
nearly all the talking, with others joining occasionally to ask something.

At 0.45 the clustering says 34 min, 14 min, 6 min, 5 min, and ten voices under a
minute each. That is the meeting. At the shipped 0.70 it says fifty-two people,
which is not.

So the doc above was right that the gap would narrow, and wrong about how much:
0.70 is not conservative on real audio, it is simply broken. One meeting is one
data point and one headset is one microphone, so this is not yet a new default —
but it is the first evidence from outside the synthesiser, and it points down by
about 0.25.

`relabel_transcript` applies a chosen threshold to an existing transcript from
the saved recording, so trying another value costs seconds rather than another
hour of recognition.
