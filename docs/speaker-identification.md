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
