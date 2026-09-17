#!/usr/bin/env python3
"""Find the words a meeting's recogniser mangled, with evidence rather than guesses.

The corrections dictionary is safe because it only ever replaces a form the user
replaced themselves. Filling it automatically would throw that away — and the
transliterating glossary that tried already turned "полка" into "Kafka", because
no phonetic threshold separates a real word from a mangled one.

A spell checker does separate them. Hunspell with the Russian dictionary accepts
"полка", "сжатие" and "индексы" and rejects "квка", "тыбаны" and "средпул". That
is the guard that was missing, and it is the only thing here that decides
anything on its own.

So this proposes and never writes:

  1. every Cyrillic token the Russian dictionary rejects,
  2. kept only if it recurs — a single odd word is a misrecognition, a word that
     comes back all meeting is a term the speakers use,
  3. handed to a local model with the lines it appeared in, to say what it
     probably should be,
  4. printed for a person to accept or throw away.

    scripts/find_terms.py <transcript.md> [--min-count 2]
"""

import json, os, re, subprocess, sys, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DICTIONARY = Path(__file__).resolve().parent.parent / "models/hunspell-ru/ru_RU"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL = os.environ.get("REPAIR_MODEL", "qwen3:8b")

# Below this a word is one bad moment rather than a term the meeting uses.
MIN_COUNT = 2
# Two letters is an initial or a filler, not a mangled name.
MIN_LENGTH = 3
# Lines of evidence shown to the model and to the reader.
CONTEXT_LINES = 3

SYSTEM = """Ты смотришь на расшифровку рабочей встречи, сделанную распознавателем речи.

Встречи идут на русском с английскими техническими терминами. Распознаватель
пишет английский термин русскими буквами: «Тыбаны» вместо Kibana, «Квка» вместо
Kafka, «сред пул» вместо thread pool.

Тебе дают одно слово, которого нет в русском словаре, и строки, где оно встречалось.

Ответь ОДНОЙ строкой в формате:
ТЕРМИН: <как это пишется по-английски>
или
НЕ ТЕРМИН

«шардов» → ТЕРМИН: shard
«консюмер» → ТЕРМИН: consumer
«дсн» → ТЕРМИН: DSN
«похоронению» → НЕ ТЕРМИН
«саша» → НЕ ТЕРМИН

Отвечай «НЕ ТЕРМИН», если это русское слово, имя человека, или если ты не
уверен. Не уверен — значит не термин. Не повторяй слово по-русски: если не
можешь назвать английское написание, это не термин."""


def lines_of(path):
    body = Path(path).read_text().split("---", 2)[-1]
    out = []
    for line in body.splitlines():
        match = re.match(r"\*\*\[[\d:]+\] [^:]+:\*\* (.*)", line)
        text = (match.group(1) if match else line).strip()
        if text:
            out.append(text)
    return out


def rejected_by_the_dictionary(words):
    """Which of these the Russian dictionary does not know."""
    result = subprocess.run(
        ["hunspell", "-d", str(DICTIONARY), "-l"],
        input="\n".join(words), capture_output=True, text=True,
    )
    if result.returncode != 0 and not result.stdout:
        raise SystemExit(f"hunspell failed: {result.stderr.strip()[:200]}")
    return set(result.stdout.split())


def ask(word, examples):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Слово: {word}\n\nСтроки:\n" + "\n".join(examples)},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
        "think": False,
    }).encode()
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        answer = json.load(response)["message"]["content"]
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
    for line in reversed([l.strip() for l in answer.splitlines() if l.strip()]):
        if line.upper().startswith("ТЕРМИН:"):
            return line.split(":", 1)[1].strip().strip('"«»')
        if "НЕ ТЕРМИН" in line.upper():
            return None
    return None


def main():
    transcript = sys.argv[1]
    minimum = int(sys.argv[sys.argv.index("--min-count") + 1]) if "--min-count" in sys.argv else MIN_COUNT

    lines = lines_of(transcript)
    counts, where = Counter(), defaultdict(list)
    for line in lines:
        for word in re.findall(r"[А-Яа-яЁё]{%d,}" % MIN_LENGTH, line):
            low = word.lower()
            counts[low] += 1
            if len(where[low]) < CONTEXT_LINES:
                where[low].append(line)

    print(f"{len(lines)} строк, {len(counts)} различных кириллических слов", file=sys.stderr)
    unknown = rejected_by_the_dictionary(sorted(counts))
    candidates = [(w, n) for w, n in counts.most_common() if w in unknown and n >= minimum]
    print(f"словарь не знает {len(unknown)}, из них повторяются {len(candidates)}", file=sys.stderr)

    proposed = []
    for word, n in candidates:
        term = ask(word, where[word])
        if term:
            proposed.append((word, term, n, where[word][0]))
            print(f"  {word} ×{n} → {term}", file=sys.stderr)

    print("# слово\tпредложено\tвстретилось\tпример")
    for word, term, n, example in proposed:
        print(f"{word}\t{term}\t{n}\t{example[:80]}")
    print(f"\nпредложено {len(proposed)} из {len(candidates)} кандидатов", file=sys.stderr)


if __name__ == "__main__":
    main()
