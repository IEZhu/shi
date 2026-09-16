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
one substantive error, `шлюзом` → `шлезом`, is domain vocabulary.

### Hotword biasing does not fix it — measured

An earlier version of this document proposed a per-meeting glossary, on the
reasoning that domain vocabulary is exactly what sherpa-onnx's hotword biasing
(`hotwords_file`, `hotwords_score`) exists for. That was a hypothesis, and
`examples/hotwords` falsified it. Biasing towards `шлюзом` on the very clip that
gets it wrong:

| score | output |
|------:|---|
| none  | …с платежным **шлезом**, нужно успеть до пятницы. |
| 1.5   | unchanged |
| 3     | unchanged |
| 6     | unchanged |
| 12    | Кто во**шлюзомшлюзомшлюзом** до пятницы. |
| 30    | Кшлюзомшлюзомшлюзомшлюзомшлюзомшлюзом… |

There is no useful window: the feature goes from no effect straight to
destroying the transcript. A glossary was therefore not built. A settings panel
that does nothing, and wrecks the output if turned up, is worse than its
absence.

Two facts fell out of the same experiment and are worth keeping:

- sherpa refuses `hotwords_file` unless `decoding_method` is
  `modified_beam_search`; with the default greedy decoding the recogniser
  simply fails to build.
- **`ё` does not exist in this model's vocabulary at all** — zero occurrences in
  `tokens.txt`, and hotwords containing it are silently skipped with
  "Cannot find ID for token ё". The `ё` → `е` in every transcript is not
  normalisation applied afterwards; the model cannot emit the letter. Nothing
  downstream can recover it, so do not spend effort trying.

Fixing domain vocabulary, if it matters later, means a different model or a
post-processing pass over the finished transcript — not biasing.

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

## A half-extracted model aborts the process

Reported from a first real run: the app crashed on "start recording", with the
stack ending in ONNX Runtime throwing out of `Ort::Session`.

The recogniser had been downloaded through the model manager, and the user
pressed start while the 464 MB archive was still being unpacked. The encoder was
405 MB of an expected 652 MB. Three things combined:

- extraction wrote straight into the directory the app looks in, so a
  half-unpacked model was indistinguishable from a finished one
- "installed" meant "the directory exists and is not empty", which a partly
  extracted model satisfies perfectly
- `OfflineRecognizer::create` does not return `None` for this failure — ONNX
  Runtime **throws**, and a C++ exception crossing into Rust aborts the process

The last one is the reason it was a crash rather than an error message, and it
cannot be caught from Rust: the prebuilt sherpa-onnx archive ships libraries
without headers, so there is nowhere to put a `try`/`catch`. The cause is
removed instead:

- archives extract to a staging directory and are moved into place with a
  rename, which is atomic
- installation finishes by writing a receipt, and only the receipt makes a model
  count as installed

The state was also unrecoverable: the truncated model still counted as
installed, so every subsequent launch crashed the same way and the app never
offered to fetch it again. A complete `.part` left by a failed install is now
recognised too, since asking a server to resume from the end of a complete file
earns a 416 rather than the bytes.

## Choosing a model for Russian and English in the same sentence

Four recognisers, the same 240 s of a real meeting — Russian speech carrying
English technical terms — plus a short clip with a known reference so the
comparison has a number in it and not only an opinion.

Speed first, on this machine, four threads:

| model | vs realtime |
|---|---:|
| NeMo fast-conformer, 10 languages | 77.6× |
| GigaAM v3 (Russian) | 20.7× |
| Parakeet TDT 0.6B v3 int8 | 14.8× |
| Whisper large-v3 int8 | **0.9×** |

Whisper large-v3 is slower than the meeting it is transcribing. An hour of call
costs an hour of machine, which rules it out for anything but a short excerpt
regardless of how well it hears.

### The mixed-language clip

A sentence of English followed by a Russian sentence with English terms in it,
generated with a known reference, so word error rate is measurable:

| model | WER | words returned | Cyrillic share |
|---|---:|---:|---:|
| **Parakeet TDT 0.6B v3** | **17 %** | 46 | 30 % |
| fast-conformer 10 languages | 42 % | 43 | 3 % |
| GigaAM v3 | 50 % | 46 | 32 % |
| Whisper large-v3 | 62 % | 21 | 64 % |

The reference is 48 words and 27 % Cyrillic. The last two columns say more than
the WER does:

- **Whisper returned 21 words of 48** — it dropped the entire English sentence
  and transcribed only the Russian. Whisper commits to one language per decoding
  window, and when a window contains two it keeps one. This is structural, not a
  tuning knob, and it applies to `whisper-turbo` as well.
- **fast-conformer returned 3 % Cyrillic** where the reference has 27 %: its
  English is excellent, and then it writes the Russian half in Latin letters —
  "Tam consumer leg weros". It never switched back.
- **GigaAM** renders English as pseudo-English: "Insuma Group Robalance Bi Is
  the Broker Lost It's Leader portition".
- **Parakeet** is the only one that put each language in its own alphabet inside
  a single utterance, and its Cyrillic share lands nearest the reference.

So the model already shipping as the default wins the case it was chosen for,
and by a wide margin. That is worth stating plainly because the measurement was
run expecting to replace it.

### Where Parakeet does lose

On Russian alone GigaAM v3 is clearly better. From the same meeting:

| | Parakeet | GigaAM v3 |
|---|---|---|
| | "было максимальное **жатие**" | "было максимальное **сжатие**" |
| | "…" (dropped) | "**Кавка**, она не могла сделать запись" |
| | "количество шардов **подкаргоин**" | "количество ша{r}дов **под каждого индекса нужно**" |

It also punctuates and marks hesitations. It is in the catalogue as
`giga-am-v3-russian` for meetings that are Russian throughout — and only those,
because the English above is what it does to the other half.

`compare_models` is the harness; point it at a WAV and a list of model
directories and it prints what each one heard.

## The newer models, and why none of them replaced Parakeet

Four months after the comparison above, three model families had appeared that
sherpa-onnx can run offline and that claim Russian and English. All three were
tested on the same clips.

| model | released | WER on the mixed clip | vs realtime |
|---|---|---:|---:|
| Parakeet TDT 0.6B v3 | 2025-08 | **17 %** | 13–15× |
| Qwen3-ASR 0.6B int8 | 2026-03 | 35 % | 5.0× |
| Omnilingual ASR 300M v2 int8 | 2026-02 | — | 6.5× |
| Nemotron 3.5 ASR 0.6B | 2026-06 | not run | — |

**Qwen3-ASR** advertises code-switching, and it is the only model in any of
these runs that spelled *Elasticsearch* correctly. It also translated the
Russian half of the clip into English — "We postponed the deployment. Some
consumer lag virus." — and on real meeting audio it invents. Among its output
for a sentence about Elasticsearch storing letters on shelves is an obscenity
nobody said, twice. That is what an LLM-based recogniser does when the audio is
unclear: it writes something fluent. For a transcript that records what
colleagues said, a confident invention is worse than a garbled word, so this one
is disqualified on a property rather than on its error rate. Raising
`max_new_tokens` from the default 128 to 1024 changed nothing — the library did
warn about truncation, and it was not the cause.

**Omnilingual ASR** (1600 languages, CTC) mixes alphabets inside single words:
"vыras", "nаgrуsкой", "sortilication nе успеваl". Unusable as it stands.

**Nemotron 3.5 ASR** is packaged only in cache-aware *streaming* form, which
needs sherpa-onnx's `OnlineRecognizer` — a code path this project does not have.
Its published numbers are good (7.91 % WER English, 9.17 % Russian at 1.12 s
chunks) and it accepts `target_lang=auto`, so it is the one worth the
integration work if the current transcript ever stops being good enough. Nothing
about its code-switching behaviour is documented, which after Qwen3 is the first
thing to measure rather than the last.

The support code for Qwen3 and Omnilingual stays: `ModelPaths` now covers both,
so trying the next release of either costs a download rather than a change.
Neither is in the catalogue, because the catalogue only offers models that were
listened to and kept.

### One bug fell out of this

`ModelPaths::verify` checked every part with `is_file`. Qwen3 ships its
tokenizer as a *directory*, so a complete model was reported as missing a file
and never reached the recogniser. Parts now declare what they are, with the
directory case pinned by a test — along with its opposite, so a tokenizer that
arrives as a plain file is still rejected.

## Can two models be combined into one better transcript?

Not usefully, and the ceiling is measurable rather than a matter of opinion.

Ten sentences were synthesised with known text — three Russian, three English,
four mixing the two the way the meetings do ("Мы посмотрели в Kibana, там
consumer lag вырос") — and a second copy was band-limited to 250–3600 Hz with
noise at 22 dB SNR to approximate a call. They live in `fixtures/mixed-ru-en`
with a manifest, so every number below is reproducible.

Three recognisers decoded each file, and `scripts/rover.py` combined them the
way ROVER does: fold each hypothesis into a word transition network by edit
distance, then vote per slot. No model is involved in the combination.

| | clean | telephone band |
|---|---:|---:|
| Parakeet alone | 19 % | 21 % |
| GigaAM alone | 36 % | 38 % |
| fast-conformer alone | 29 % | 35 % |
| ROVER, flat vote | 19 % | 21 % |
| ROVER, weighted by alphabet | 19 % | 21 % |
| **best possible from these three** | **16 %** | **17 %** |

The last row is the per-word oracle: align the reference into the network and
count a word correct if *any* system produced it. No voting rule can beat it.
Three points of headroom, and voting captured none of them, because the errors
are correlated — the three models fail on the same words.

Which words is not a mystery:

| | Russian | English | mixed |
|---|---:|---:|---:|
| Parakeet | 4 % | 6 % | **38 %** |

Russian is fine. English is fine. Every remaining error is an English technical
term inside Russian speech.

### The glossary that followed, and why it is not in the product

If the errors are named entities, replace them afterwards: transliterate each
Cyrillic token to Latin letters and match it against a list of terms the
meetings use. `scripts/glossary.py` does that, and it repairs real cases —
"тыбаны" → Kibana, "Кавка" → Kafka, "консьюмер" → consumer.

It also turns "полка" into "Kafka". After phonetic folding the two are the same
distance apart as the good matches are, so no threshold separates them, and
"полка" is a word these meetings genuinely use — the hot shelves an index is
stored on. Measured:

| | clean | telephone band |
|---|---:|---:|
| ROVER | 19 % | 21 % |
| \+ glossary everywhere | 24 % | 26 % |
| \+ glossary only where the models disagree | 18 % | **25 %** |

Restricting repairs to slots where the recognisers disagree — using the ensemble
as an uncertainty detector rather than as a chooser, which is the one thing the
oracle says it is good for — helps slightly on clean audio and hurts on the
audio that matters. The guard that would fix this is a Russian wordlist to
protect real words from being "repaired"; this machine has none, and tuning
thresholds harder on a hundred-word corpus would be fitting noise.

Both scripts stay as instruments. Neither is wired into the app.

### What has headroom

Not combination. The single-model number is the thing to move, and the only
candidate with published evidence of doing so is Nemotron 3.5 ASR: 7.91 % WER on
English and 9.17 % on Russian at 1.12 s chunks, with `target_lang=auto`. It
needs an `OnlineRecognizer` path, which this project does not have — and its
code-switching behaviour is undocumented, so that is the first thing to measure
rather than the last.

## Nemotron 3.5, and the option that is not there yet

The integration was built: `ModelPaths::StreamingTransducer`, a
`StreamingTranscriber` over sherpa-onnx's `OnlineRecognizer`, and
`load_recognizer` to route a directory to whichever decoder it needs. A
streaming model is decoded one whole utterance at a time — the voice-activity
detector has already closed the utterance, so partial results are of no use
here, and the chunking stays inside the library.

Two things had to be right.

**Padding.** A cache-aware model spends its first chunk warming its cache and
needs another to push the last of the audio through. Handed a bare utterance it
returned

```
Broker lost its leader partition and the consumer group
```

for a sentence that began "The" and ended "rebalanced" — clipped at both ends.
One chunk of silence each side fixes it exactly, and with it the same sentence
comes back word for word:

```
The broker lost its leader partition and the consumer group rebalanced
```

`tests/streaming.rs` pins that, and skips when the model is not installed.

**The language, which cannot be set.** The model's own README says to "use
per-stream language strings such as `en`, `ja`, or `auto`", and NVIDIA's
published figures — 7.91 % English, 9.17 % Russian — are quoted *with language
input*. sherpa-onnx exposes `SherpaOnnxOnlineStreamSetOption` for this. Asked
whether it knows any of twenty plausible keys — `language`, `target_lang`,
`lang`, `prompt_index` and the rest — this build answers no to all of them:

```
options this model admits to knowing:
  none of 20 candidates
```

`SetOption` is still an open feature request upstream, and 1.13.6 is the newest
crate published, so there is no version to move to. The model therefore runs in
whatever mode it defaults to.

Measured that way, on the same corpus:

| | overall | ru | en | mix |
|---|---:|---:|---:|---:|
| Parakeet TDT 0.6B v3 | **19 %** | 4 % | 6 % | 38 % |
| Nemotron 3.5 (clean) | 29 % | 17 % | 9 % | 52 % |
| Parakeet, telephone band | **21 %** | 8 % | 9 % | 38 % |
| Nemotron 3.5, telephone band | 36 % | 33 % | 12 % | 57 % |

Its English is close to Parakeet's — 9 % against 6 % — and its Russian is four
times worse. That shape is what a model defaulting to English looks like, which
fits the missing selector rather than contradicting the published numbers. It
is not evidence that the numbers are wrong; it is evidence that they are out of
reach from Rust today.

So the default stays Parakeet, and the streaming path stays in the codebase
with `examples/probe_options` next to it. The day a sherpa-onnx release
registers a language option, this becomes a download and one string rather than
an integration.

## Levelling the audio before the recogniser sees it

`preprocess.rs` removes any constant offset, and can bring an utterance to
0.08 RMS with the gain capped and the peak protected from clipping. Levelling
is **off by default**, and the reason is the more useful half of this section.

On a corpus scaled down to imitate a faint talker it is a large win:

| corpus | untouched | levelled |
|---|---:|---:|
| telephone band, RMS 0.02 | 21 % | 21 % |
| quiet, RMS 0.002 | 21 % | 19 % |
| very quiet, RMS 0.0005 | **43 %** | **24 %** |

Run against the first real meeting, it made the transcript worse.

### The statistic that misled me

The ceiling was first set to 40 dB from that recording's microphone stream,
which measured between 0.001 and 0.009 RMS per minute. Those minutes were mostly
silence. Measured per *utterance* instead, the two moments the speaker actually
spoke sat at **0.10 and 0.16** — above the target, needing no gain at all.
Everything else was breath and room between 0.00006 and 0.008.

So the microphone was never quiet. The person was.

Levelled, the recogniser produced thirteen lines it had previously left empty:
"Thank you." five times, "Yeah.", "I'm just gonna be able to do" — English
filler in a Russian meeting, which is what a recogniser writes over breath.
Thirteen invented lines, none recovered.

### Why the gain ceiling is not the control

Lowering it to 20 dB produced **twenty-five** such lines where 40 dB produced
twenty-three. That rules out amplification as the mechanism.

The voice-activity detector sees the raw audio, so segmentation is identical in
every run; what changes is how many segments come back with text instead of
nothing. Those segments hold no speech. An empty transcript was the only thing
catching them, and levelling takes that away.

Fixing this properly means telling a faint talker from a quiet room before
deciding whether to apply gain. Until that is measured, the stage stays off,
`preprocess_with` switches it on, and a test pins that it works when it is.

### The denoiser is off too

GTCRN over the same utterances: 25 % and 26 % against 21 % and 19 % without it.
A model trained to please the ear removes the low-energy detail a recogniser
leans on. `denoise_with` attaches one so the next can be measured rather than
argued about.

### Why the language is not chosen per utterance

Parakeet is best or tied in every category on this corpus except Russian through
a telephone band, where GigaAM scores 4 % against its 8 %. That is one third of
the material and four points, against a second model resident in memory and a
language detector in front of it. The measurement is here if the balance changes.

## Two monolingual recognisers, spliced

The proposal: stop asking one multilingual model to handle both languages. Run
a Russian specialist and an English specialist over the same audio and merge the
two transcripts, taking each word from whichever model owns its language.

This is not the ROVER experiment above. Voting asks which hypothesis is most
popular and cannot help when every system is wrong on the same word. Splicing
asks which recogniser is competent for the word in front of it, which is a
different question and deserved its own measurement.

Until now the pool held no monolingual English model at all — Parakeet v3,
fast-conformer and Whisper are all multilingual, and GigaAM is Russian-only. So
**Parakeet TDT 0.6B v2 int8** was added: the English-only sibling of the model
already in use, same architecture, same file layout, top of the Open ASR
leaderboard for English. `scripts/splice.py` does the merging and the scoring.

| | overall | ru | en | mix |
|---|---:|---:|---:|---:|
| Parakeet v3 (multilingual) | **19 %** | 4 % | **6 %** | 38 % |
| Parakeet v2 (English only) | 71 % | 100 % | 9 % | 105 % |
| GigaAM v3 (Russian only) | 36 % | 4 % | 53 % | 40 % |
| splice by alphabet | 20 % | 4 % | 6 % | 40 % |
| splice where the models disagree | 20 % | 8 % | 6 % | 38 % |
| **best conceivable selection** | **15 %** | 0 % | 6 % | **31 %** |

### The English specialist is not better at English

Nine per cent against six is three errors against two, out of thirty-three
words. On a corpus this size that is not a difference. What matters is *which*
words: both models mangle "Elasticsearch", and they mangle it almost identically
— "Elastic Sark" against "ElasticSark" — on clean English audio with no Russian
anywhere near it.

That single observation is what decides the whole idea. A specialist is only
worth its memory if it is better on its own language, and this one is not.

### The ceiling says no rule exists

The last row is the oracle: take the reference word whenever *any* of the three
produced it. No selection rule, however clever, can beat it. Adding the English
specialist moves it from 16 % to 15 % overall, and from 33 % to 31 % on the
mixed sentences that hold all the error. One point, two on mix.

Both concrete rules land above Parakeet alone. Gating on disagreement — using
the two Russian-capable models as an uncertainty detector, the one thing the
oracle says an ensemble is good for — holds the mixed sentences at 38 % and
loses four points on Russian, because Parakeet and GigaAM also disagree on
Russian words the English model then overwrites with Latin guesses.

### Why cutting the recognised speech out of the audio does not rescue it

The stronger version of the proposal: decode one language, remove what was
recognised from the waveform, hand the remainder to the other model with less
room to go wrong. The oracle above does not bound this, and saying so honestly
matters — a cascade changes what the second model *hears*, so it can produce
hypotheses that are not in the network at all.

It fails on its premise instead. The second stage would be a specialist that is
not better on its own language, and "Elasticsearch" is already wrong with zero
Russian to remove. Cutting away Russian cannot fix an error that happens when no
Russian is present.

The mechanics are also weaker than they sound. A recogniser returns text and
timings, not the waveform of the words it heard, so "remove what was recognised"
can only mean blanking time spans. That needs word-level timestamps *and* a
decision about which spans are foreign — and that decision is the disagreement
detector, measured at 38 % on the mixed sentences. It does not find them.

### Latin hotwords fail the same way Russian ones did

Biasing was falsified earlier on a Russian word. The failing terms here are
Latin, and Latin tokens certainly exist in the vocabulary, so it was worth one
more run — on `06-mix.wav`, biasing towards Kibana, Elasticsearch, consumer, lag:

| score | output |
|------:|---|
| none | …там **консумер Лэк** вырос, и **Лэстиксок** начал отдавать 429. |
| 1.5, 3, 6 | unchanged |
| 12 | …там консумер **laglag** вырос, и **ElagSlag** начал отдавать 429. |
| 30 | Мы посмотрели В**laglaglaglaglaglag**… |

No window, exactly as before. The script was never the problem.

### What the errors actually are

Kibana, Kafka, Elasticsearch, consumer lag, shard allocation, rebalance, thread
pool, consumer offsets topic. A closed set of named entities, mangled the same
way by every model tested, in the same places.

That is not a language-selection failure, and three experiments aimed at
language selection — routing, voting, splicing — have now each been stopped by
the same wall. It is a vocabulary failure, and a word-level decision made
without the sentence around it cannot fix it: "полка" and "Kafka" are the same
distance apart as the repairs that work. The context that separates them exists
only in the finished transcript.

## The dictionary the user teaches

Three experiments aimed at language selection failed against the same wall, and
the wall is a closed set of named entities every model mangles identically. The
context that would separate "полка" from "Kafka" does not exist at the moment a
recogniser emits a word — but it does exist in the finished transcript, in front
of the person who was in the meeting.

So the transcript is editable, and an edit is read for what it teaches. Correct
one line and the term is repaired in every other line, in every past meeting,
and in the ones still to come. `corrections.rs` holds the mechanism.

Nothing is guessed. A form is replaced only if the user replaced it themselves
at least once, which is the guard the transliterating glossary never had.

### What a rule is keyed on

Not the phonetics, and not the word boundaries.

The failed glossary matched by transliterating Cyrillic and comparing — and
after phonetic folding "полка" sits as close to "Kafka" as the repairs that
work. Here the key is the letters the recogniser committed to, folded to lower
case with `ё` mapped to `е`, which it cannot emit anyway.

Word boundaries are dropped from the key, and that is a measurement rather than
a convenience. The same term came back as "сред пул" on clean audio and
"средпул" through a telephone band; "тайм-аут" and "таймаут" appear in the same
corpus. Where a transducer puts a space is a guess it makes differently every
time. The letters are steady, so one rule covers every spelling of the gap. A
comma is different — that is a claim about the sentence, not a guess about a
boundary — so a phrase never matches across one.

### It transfers to recordings it never saw

The claim worth testing is that a repair taught once keeps working. Determinism
is only partial: on the degraded corpus Parakeet wrote "КБ" where the clean
audio gave "Тыбаны", and no dictionary can bridge that.

`examples/dictionary_transfer` measures what does survive. It corrects the four
mixed sentences of one corpus, then applies the rules to the *other* corpus and
scores it.

| taught on | applied to | overall | ru | en | mix |
|---|---|---:|---:|---:|---:|
| clean | degraded | 21 % → **18 %** | 8 % → 8 % | 9 % → 9 % | 38 % → **31 %** |
| degraded | clean | 19 % → **15 %** | 4 % → 4 % | 6 % → 6 % | 38 % → **29 %** |

Russian and English are untouched to the point. That is the check the glossary
failed, and it passes here because nothing fires on a word the user did not
personally correct.

Worth putting beside the ensemble numbers: the best conceivable selection from
three recognisers — the oracle no voting or splicing rule can beat — was 15 % on
the clean corpus and 18 % on the degraded one. The dictionary reaches both,
from a single model, on audio it was not taught on.

### What it does not do

A rule fires on the letters it was taught. When the recogniser mangles a term
into something else entirely, the rule misses, and the transcript is exactly as
wrong as before — no worse. That is the honest shape of the feature: it never
guesses, so it never damages, and its coverage grows only as fast as the user
corrects things.

`hits` is stored per rule and shown in the dictionary panel for that reason. A
rule that has never fired is a rule that was taught on a form the recogniser has
not produced since, and the user can drop it.

## A voice gate in front of the recogniser

Over the first real meeting the voice-activity detector opened **277 segments
on the microphone — 49.4 of 68 minutes — for a person who spoke three times.**
The system stream, which carried the conversation, gave 196 segments and
61.9 minutes. What filled the microphone segments was breath and room tone, and
the only thing keeping it out of the transcript was that the recogniser usually
returned nothing for it. Levelling showed how thin that protection was.

Loudness could not be the test — a faint talker sits where a quiet room sits —
so the gate asks whether anything in the segment repeats itself the way a voice
does: normalised autocorrelation at lags between 70 and 400 Hz, which is the
same for a voice recorded faintly as for one recorded loudly. `speech.rs`.

### A run, not a share

The first design scored the *share* of voiced frames and set the bar at 0.05.
Measured stand-alone it looked safe: the least periodic of 165 real remote
turns scored 0.247, the microphone's three real turns 0.221–0.568, and segments
that produced text out of nothing 0.000–0.006. The share is diluted by whatever
silence the segment contains, though, and a detector segment is mostly silence
by nature. So the gate measures the **longest unbroken stretch of voicing**
instead:

| | shortest run inside real speech | median run |
|---|---:|---:|
| 165 remote turns | **190 ms** | 800 ms |
| 3 microphone turns | 420 ms | 470 ms |
| segments that produced text out of nothing | — | 80 ms |
| silent segments | — | 50 ms |

The bar is **100 ms**: half the shortest real turn seen, still removing three
quarters of the silent segments. The margin is on the side of keeping speech.

### Measured through the pipeline, over the whole meeting

| | segments rejected | decoding avoided | words lost |
|---|---:|---:|---|
| microphone | 195 | **31.4 of 49.4 min** | 8, all filler — "mm", "yeah", "okay", "thank you" |
| system | 9 | 0.2 min | 5 — "Oh", "Next.", "S.", "Thank you." |

Every one of the nine system segments was checked against what the ungated run
had heard at that moment: those four fragments and six empties. The three real
microphone turns and all 7 115 words of the remote side came through — 7 110
after the five above. The A/B was run twice with the identical binary and is
byte-for-byte reproducible.

### The measurement that said otherwise, and was wrong

An earlier comparison reported the gate dropping 103 and then 344 real words
from the system stream, which nearly killed it. That number came from a script
that read only the first line of each Markdown block. Blocks hold one utterance
per line, and the gate changes speaker labels downstream — a rejected segment
alters the tracker's state — so the *block count* fell from 136 to 119 and the
script counted fewer first lines. Counting every line: 7 115 against 7 110.

Two lessons, both already learned once and now learned again:

- **count the whole file**, not the shape of it;
- **the stand-alone number was right** and the pipeline number was an artefact,
  which is the opposite of what happened with levelling — so neither "trust the
  harness" nor "trust the pipeline" is a rule; check the measurement itself.

### The detector's threshold is not the lever

Silero's threshold had never been swept on real audio; 0.5 came from the plan.

| threshold | microphone | system |
|---:|---|---|
| 0.5 | 277 segments, 49.4 min | 196, 61.9 min |
| 0.6 | 287, 45.4 min | 196, 61.7 min |
| 0.7 | 293, 40.3 min | 201, 61.5 min |
| 0.8 | 294, 33.1 min | 206, 61.2 min |

Raising it *splits* the noise segments rather than removing them: the count
goes up while the minutes go down. Silero scores this room above 0.8. The gate
stays.

Fed the same four seconds of that room on their own, Silero opens **no segment
at all**; it opened one only with an hour of meeting behind it. That is why no
fixture ever caught the problem, and why `fixtures/room-tone-mic.wav` is tested
against the gate directly rather than through the detector.

### The detector's version, on the other hand

`models/silero_vad.onnx` — the 2025-07 export, 644 KB — segments the meeting
exactly as `silero_vad_v4.onnx` does: 277 and 196 segments, to the second.
**Silero v5** is a different animal:

| model | microphone | system |
|---|---|---|
| current (= v4) | 277 segments, 49.4 min | 196, 61.9 min |
| **v5** | **6 segments, 0.3 min** | 162, 62.7 min |

Six microphone segments: two empties, one "Yeah.", the first and third real
turns intact — and the second real turn **cut from 3.8 s to 1.8 s** and
returned as "But the light." One of three real turns damaged is exactly the
failure the gate was designed never to commit, so v5 is not the default. It is
in `models/` as an instrument; the pipeline comparison follows below.

### v5 through the pipeline: a latent abort, not a comparison

The pipeline pass with v5 never finished. At the fiftieth minute the detector
had one speech segment open for over two minutes — sherpa reports its circular
buffer overflowing at 1 920 000 samples and grows it — and then handed that
segment to the recogniser, whose encoder cannot take an input that long. ONNX
Runtime threw (`Attempting to broadcast an axis by a dimension other than 1.
12288 by 14321`) and the process aborted, because a C++ exception crossing into
Rust cannot be caught. `max_speech_duration` was set to 20 s throughout and did
not close the segment; the 22–29 s segments seen earlier with v4 were the same
thing on a smaller scale.

So this is not only about v5. **Any segment the detector fails to close will
crash the app mid-meeting**, and the shipped detector merely has not produced
one yet on the recordings tried. The recogniser must never be handed more than
it can take, whatever the detector does; that guard belongs in the pipeline,
and until it exists v5 cannot even be measured here.

### The recogniser is now capped — and the cap turns out to fire

`split_utterance` in `engine.rs` hands the recogniser nothing longer than
`VadSettings::longest_decode` — 30 s by default. A longer utterance is cut at
the quietest 20 ms in the two seconds before the limit, so a pause is preferred
to the middle of a word, and each piece is judged, decoded and attributed on its
own. `the_recogniser_is_never_handed_more_than_it_can_take` pins it through the
real detector with a small cap.

It was meant as a safety net that never fires with the shipped detector, and
that was wrong. The 29 s overshoot quoted above was the *microphone*; the
system stream of the same meeting holds segments up to **74 s**, forty-five of
them past 30 s, and `max_speech` at 20 s stops none of them. Over the meeting
the cap cut about 46 times, and **98 of the 230 finals were longer than
20 s**. The transcript changed by some 300 words either way (7 115 → 7 199, a
net gain), which is not the boundary effect one would expect from 46 cuts. The
section after the next explains why.

Natural pauses do not solve this. Lowering the detector's `min_silence` moves
the median utterance but leaves the tail exactly where it was:

| `min_silence` | system: median | > 20 s | > 30 s | longest |
|---:|---:|---:|---:|---:|
| 500 ms | 14.3 s | 84 | 45 | 74.2 s |
| 300 ms | 10.0 s | 81 | 43 | 74.2 s |
| 200 ms | 9.1 s | 74 | 39 | 74.2 s |
| 120 ms | 7.2 s | 69 | 38 | 74.2 s |

A fluent speaker on a call does not leave a 120 ms gap the detector notices, so
the long stretches can only be cut by the pipeline — which is what the cap does,
and why its length is a setting.

### v5 through the guarded pipeline

With the cap in place v5 runs to the end: 216 finals, buffer overflow warnings
but no abort, the remote side at 7 252 words against v4's 7 199 with the same
churn of a few hundred words either way. The microphone tells the same story
as the stand-alone run: eleven filler lines become two, the first and third
real turns survive, and the second is still cut to 1.8 s and returned as
"But the light." — at thresholds 0.5, 0.4 and 0.3 alike, so it is not the
threshold. Two suppressed echoes become none, because v5 never opened on them.

One damaged turn in three is the failure this whole section exists to avoid,
and the voice gate now removes the microphone noise without touching a word, so
v4 stays. v5 remains in `models/` for the day a recording contradicts this.

### The length of an utterance is the largest lever found so far

The churn from 46 cuts was too large to be a boundary effect, so the effect of
piece length was measured directly. The ten reference sentences of
`fixtures/mixed-ru-en` were concatenated with 300 ms gaps into 43 s of speech,
in four different orders so that where the cuts fall averages out, and decoded
whole, in halves, in quarters — every cut in a gap — and sentence by sentence.
About 400 reference words per cell:

| | one piece, ~43 s | two, ~22 s | four, ~11 s | sentence by sentence, ~4 s |
|---|---:|---:|---:|---:|
| clean | 32 % | 33 % | 29 % | **18 %** |
| telephone band | 46 % | 40 % | 36 % | **22 %** |

The same audio, the same model. **Decoding it in long stretches roughly
doubles the error rate.** Even eleven-second pieces sit ten points above
single sentences. The whole-piece decode of the clean set dropped "lost its
leader partition" from the middle of an English sentence it decodes perfectly on
its own. A single cut placed deliberately mid-word at 12 s still scored better
than no cut at all (26 % and 42 % against 27 % and 49 %), so the damage a cut
does is smaller than the damage length does.

Nothing about the model was changed in any of the six experiments that tried to
replace it, and the largest single number in this document was sitting in the
detector's `min_silence`: at 500 ms a fluent speaker is never interrupted, so
the recogniser is fed twenty- and thirty-second stretches it demonstrably
cannot handle. On the real meeting 98 of 230 finals were over 20 s.

What this does not settle is how short. The corpus was cut in gaps that exist
because the sentences were recorded separately; a real monologue has to be cut
by energy at the quietest moment available, which is what the cap does, and
`min_silence` has just been shown not to help. The measurement that decides is
the same 71-minute meeting decoded with `longest_decode` at 10 s instead of 30
(`SHI_LONGEST_DECODE_MS=10000` on `transcribe_recording`) — and, since it has no
reference transcript, a person who was there reading the two side by side.

### Ten seconds, on the meeting

`SHI_LONGEST_DECODE_MS=10000` over the same recording: **488 finals instead of
246**, median 8.7 s, none over 10 s. The remote side came back with 7 357 words
against 7 199 — 574 gone, 732 new — which is the direction the corpus predicts,
since a long decode loses words rather than inventing them. The microphone was
untouched at 51.

One passage read side by side shows both effects at once. The shorter version
keeps the fillers the speaker actually said ("да", "как бы") that the long
decode had smoothed away, and it mangles the one phrase the cut landed inside:
"было принято решение посмотреть в эту сторону" became "было пришел посмотреть
в сторону". The quietest 20 ms in a two-second window is not always a pause.
Which effect outweighs which is a judgement for someone who was in the room;
both transcripts sit in `target/dev-bundles/` as `v4guard.md` and `cap10.md`.

What is not a judgement: at the shipped session threshold of 0.70, twice as
many utterances turned into **108 speakers** after offline clustering instead of
58. Decode length and the diarization threshold are coupled — shorter pieces
give the tracker less voice per decision — so the threshold has to move to the
measured 0.45–0.50 before a shorter decode can be shipped, whatever the reader
decides.

## The length of a decode is the setting that mattered

Six experiments tried to replace the model and none of them beat it. The lever
was never which model but how much audio it is asked to hear at once.

`examples/decode_length` runs the shipped pipeline — the real detector, the real
`split_utterance` cutting at the quietest moment it can find, which is sometimes
inside a word — over the reference corpus at several caps. Four sentence orders
with gaps varying between 200 and 600 ms, so no cap can line up with a rhythm;
about 400 reference words a cell:

| cap | clean | telephone band |
|---:|---:|---:|
| 5 s | **26 %** | **30 %** |
| 8 s | 26 % | 34 % |
| 12 s | 38 % | 42 % |
| 30 s | 38 % | 47 % |

Below five seconds it turns again — three-second pieces score 30 % and 31 % on a
single order — because the cut lands inside a word too often. The mechanism is
visible in the word count rather than the error rate: at thirty seconds the
recogniser returns between 48 and 99 words of a hundred, at five seconds
between 92 and 103. A long decode does not garble, it **drops**.

`longest_decode` is therefore five seconds.

### What that broke, and what it taught

Five-second pieces made the transcript better and the speaker labelling far
worse: the same meeting went from 58 speakers to 144 on the online pass.

The recogniser and the voice embedder want opposite things. A decode is more
accurate the shorter it is; an embedding is more accurate the longer it is.
Splitting for the recogniser had been splitting for both.

So the two were separated. The speaker is decided once, from the whole
utterance, and the pieces inherit that one verdict — lazily, so an utterance
that turns out to be all echo or all room tone never reaches the tracker. The
offline pass joins the pieces back before embedding: they abut exactly, and the
join is one-sided so a span that *overlaps* its neighbour is left alone.

| | blocks | remote words | speakers |
|---|---:|---:|---:|
| 30 s, fixed 0.70 | 148 | 7 199 | 57 |
| 5 s, derived threshold | 173 | 7 370 | 39 |
| **5 s, voice judged whole** | **84** | **7 370** | **17** |

903 pieces rejoin into 185 utterances, against the 184 the old segmentation
produced — which is the check that the rejoining is reconstructing what was
split rather than inventing groups.

## A threshold the meeting derives for itself

The session threshold was 0.70, from a synthetic corpus, and a real meeting
wanted 0.45. Both numbers are right about their own recording, and a fixed one
cannot serve both: on the calibration corpus the closest pair of *different*
voices scores 0.662, so a tracker at 0.45 merges two people —
`the_hardest_pair_is_still_separated` fails the moment it is tried.

What survives both is the shape. Pairwise similarities are bimodal: a crowded
low mode of different speakers, a smaller high mode of one speaker heard twice.
`valley_threshold` finds the split between them by Otsu's method, which has no
constant to tune.

Asked of the real meeting it answered **0.44** on five-second pieces — within a
hundredth of the 0.45 that had been found by sweeping the threshold by hand —
and **0.50** once the pieces were rejoined and the embeddings improved. The
online pass keeps the fixed 0.70, where there is no distribution to look at yet
and merging two people is the unrecoverable error.

The result reads like the meeting its participant described, "about fourteen of
us, two or three did nearly all the talking". Words per speaker:

```
4409, 1374, 645, 525, 180, 70, 63, 47, 27, 10, 5, 5, 4, 2, 2, 1, 1
```
