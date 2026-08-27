# Transcription — measured, not assumed

The plan made real-time factor a gate rather than a parameter, because the
published Parakeet figures could not be applied to this workload. They come
from batched GPU runs; the pipeline decodes one short utterance at a time on a
CPU while a meeting is happening.

## What the model actually costs here

Apple silicon, 14 cores, `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`,
averaged over four clips, first pass discarded:

| Threads | RTFx | Where `Cadence` settles |
|--------:|-----:|---|
| 1  | 3.3  | draft every 1.6 s, window 2.6 s |
| 2  | 6.2  | draft every 800 ms, window 2.5 s |
| 4  | 11.1 | draft every 800 ms, window 4.5 s |
| 8  | 18.2 | draft every 800 ms, window 7.3 s |

The headline RTFx for this model is above 3000. Here it is **3.3 to 18.2** —
two to three orders of magnitude lower. Anything built on the published number
would have been wrong by that margin, which is why the draft cadence is a
controller and not a constant.

Russian clips measured slightly faster than the European set (13.8 at four
threads), so the table above is the conservative case.

## Why the cadence is a controller

[`Cadence`](../crates/pipeline/src/cadence.rs) measures decode cost per second
of audio and spends at most half of each interval on drafts. Given the numbers
above it converges to:

- a fast machine: short interval, long window — drafts feel live
- a slow machine: longer interval, shorter window — drafts thin out

Degradation is gradual and is pinned by a test: drafts must never collapse to
nothing, and quality must fall monotonically with speed rather than off a cliff.

## Accuracy

Parakeet handles Russian, which was the open question. Synthesised speech, so
treat this as a smoke test rather than a WER measurement:

> spoken: «Кто возьмёт задачу по интеграции с платёжным шлюзом? Нужно успеть до пятницы.»
> decoded: «Кто возьмет задачу по интеграции с платежным шлезом, нужно успеть до пятницы.»

Punctuation, capitalisation and per-token timestamps all arrive for free. The
one substantive error, `шлюзом` → `шлезом`, is domain vocabulary — precisely
what sherpa-onnx's hotword biasing (`hotwords_file`, `hotwords_score`,
`create_stream_with_hotwords`) exists for. A per-meeting glossary of names and
product terms is the natural fix, and is worth doing before chasing a better
model.

`ё` is normalised to `е` throughout, which is standard for ASR output and not
worth correcting.

## Drafts need no voice-activity gate of their own

Whisper invents text during silence; Parakeet does not. That means a draft
decode over a window that turns out to be silence simply returns nothing, so
the draft path can re-decode the open utterance without a separate speech gate.
The VAD still decides where utterances begin and end — it is just not on the
critical path for drafts.

## The echo problem, reproduced

Running a full meeting through the built app with speech played over the
built-in speakers produced this:

```
**[15:27:05] Собеседник:** Вчера мы выкатили релиз метрики ровные.
**[15:27:08] Вы:**         Вчера мы выкатили релиз, не поли.
```

Every utterance appears twice. The system stream captured it digitally; the
microphone picked the same sound out of the air a beat later. The transcript
doubles, the second copy is worse, and a diarizer would conclude the local user
said everything the remote side did.

Three properties of the duplicate make it detectable:

- it lags the original by the speaker-to-microphone flight time, well under 200 ms
- it is strongly correlated with the system stream over that lag
- it is quieter and band-limited compared to the direct capture

So suppression is a cross-correlation between the two streams rather than
anything model-shaped. Headphones make the problem disappear entirely, which is
why the readiness panel should say so when the output device is built-in —
`cpal` reports `InterfaceType::BuiltIn` versus `Bluetooth`/`Usb` for free.

This is why the plan put echo handling in the MVP rather than in polish: without
it the app is unusable in the configuration most people will first try.
