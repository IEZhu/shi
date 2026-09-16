#!/usr/bin/env python3
"""Repair a transcript line by line with a local language model.

Measured, not assumed. Three experiments aimed at choosing between hypotheses —
voting, splicing, per-utterance language routing — were each stopped by the same
ceiling: the oracle over the N-best list, 15 % on this corpus. No selection rule
beats it, because the right word is in none of the hypotheses.

A generative corrector is not bounded by that oracle; it can write a word no
hypothesis contained (HyPoradise, NeurIPS 2023). The published work closest to
this problem — bilingual code-switched speech — reports that a dictionary on top
of an LLM beats the LLM alone, and the dictionary already exists here.

The danger is the one Qwen3-ASR was rejected for: a model that writes something
fluent when it is unsure. So the corrector is fenced in:

  * one utterance at a time, never the whole transcript, so it cannot invent a
    narrative;
  * the dictionary's terms are handed over as the only vocabulary it may
    introduce;
  * anything that drifts too far from what was heard is thrown away and the
    original kept, because a garbled word is better than a confident invention.

    scripts/llm_repair.py <hypotheses.tsv> <out.tsv> [--terms t1,t2,...]
"""

import json, os, re, sys, urllib.request
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL = os.environ.get("REPAIR_MODEL", "qwen3:8b")

# How far a repaired line may drift from the original, as a share of its words.
# Fixing named entities changes a word here and there; rewriting a sentence
# changes most of it, and that is the failure this guards against.
MAX_DRIFT = 0.34

SYSTEM = """Ты правишь расшифровку встречи, сделанную распознавателем речи.

Встречи идут на русском с английскими техническими терминами. Распознаватель
часто пишет английский термин русскими буквами: «Тыбаны» вместо Kibana, «Квка»
вместо Kafka, «сред пул» вместо thread pool.

Правила, обязательные:
1. Верни ту же реплику, исправив только искажённые термины.
2. Не добавляй, не убирай и не переставляй ничего, кроме этих терминов.
3. Не дописывай продолжение и не объясняй ничего.
4. Если не уверен — оставь как есть. Оставить ошибку лучше, чем выдумать.
5. Ответ — только исправленная строка, без кавычек и комментариев."""


def words(text):
    return re.sub(r"[^\w\s]", " ", text.lower().replace("ё", "е"), flags=re.UNICODE).split()


def drift(before, after):
    """Share of words that changed, by edit distance over word sequences."""
    a, b = words(before), words(after)
    if not a:
        return 0.0 if not b else 1.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[len(b)] / len(a)


def ask(line, terms):
    prompt = line if not terms else (
        f"Термины, которые встречаются в этих встречах: {', '.join(terms)}.\n\n"
        f"Реплика:\n{line}"
    )
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # Greedy: a transcript wants the likeliest reading and has to be
        # reproducible from one run to the next.
        "options": {"temperature": 0.0, "num_predict": 512},
        "think": False,
    }).encode()

    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        answer = json.load(response)["message"]["content"]
    # Some models narrate their reasoning first; keep the last non-empty line.
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
    lines = [l.strip().strip('"«»') for l in answer.splitlines() if l.strip()]
    return lines[-1] if lines else line


def main():
    source, target = sys.argv[1], sys.argv[2]
    terms = []
    if "--terms" in sys.argv:
        terms = [t.strip() for t in sys.argv[sys.argv.index("--terms") + 1].split(",") if t.strip()]

    out, kept, changed, refused = [], 0, 0, 0
    for row in Path(source).read_text().splitlines():
        if "\t" not in row:
            continue
        name, line = row.split("\t", 1)
        if not line.strip():
            out.append(row)
            continue
        try:
            repaired = ask(line, terms)
        except Exception as error:
            print(f"  {name}: запрос не удался ({error}), оставляю как есть", file=sys.stderr)
            out.append(row)
            continue

        moved = drift(line, repaired)
        if moved > MAX_DRIFT:
            print(f"  {name}: отклонено, сдвиг {moved:.0%}\n     было:  {line}\n     стало: {repaired}",
                  file=sys.stderr)
            refused += 1
            repaired = line
        elif moved > 0:
            changed += 1
        else:
            kept += 1
        out.append(f"{name}\t{repaired}")

    Path(target).write_text("\n".join(out) + "\n")
    print(f"{target}: исправлено {changed}, без изменений {kept}, отклонено {refused}", file=sys.stderr)


if __name__ == "__main__":
    main()
