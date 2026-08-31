#!/usr/bin/env python3
"""Combine several ASR hypotheses into one, by alignment and voting.

ROVER (Fiscus, 1997): align every system's word sequence into a shared word
transition network, then pick each slot by vote. No model is involved in the
combination — it is edit distance and counting.

With three systems a majority exists, but plenty of slots still tie two-to-one
against a null, or split three ways. Those are decided by which system is known
to be competent for the script of the word in question, which is the one piece
of domain knowledge here and comes from measurement, not taste:

    GigaAM is Russian-only and renders English as pseudo-English.
    fast-conformer writes Russian in Latin letters once it has switched.
    Parakeet keeps both alphabets but is the weakest of the three on Russian.
"""

import json, re, sys
from pathlib import Path

CYRILLIC = re.compile(r'[Ѐ-ӿ]')
LATIN = re.compile(r'[A-Za-z]')


def words(text):
    text = text.lower().replace('ё', 'е')
    return re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE).split()


def script_of(word):
    cyr = len(CYRILLIC.findall(word))
    lat = len(LATIN.findall(word))
    if cyr > lat:
        return 'cyr'
    if lat > cyr:
        return 'lat'
    return 'other'


# How much each system's vote is worth for a word of each script.
# 1.0 is a plain vote; the deltas are the measured competences.
WEIGHTS = {
    'parakeet': {'cyr': 1.2, 'lat': 1.4, 'other': 1.2},
    'gigaam':   {'cyr': 1.6, 'lat': 0.4, 'other': 1.0},
    'fastconf': {'cyr': 0.7, 'lat': 1.3, 'other': 1.0},
}
PLAIN = {name: {k: 1.0 for k in ('cyr', 'lat', 'other')} for name in WEIGHTS}


def align_into(slots, hypothesis, system_count):
    """Fold one hypothesis into the network, returning the new slot list.

    Slots hold one entry per system seen so far; a system that said nothing
    where others spoke gets None, which is a vote for deleting the word.
    """
    n, m = len(slots), len(hypothesis)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + 1        # slot with nothing aligned to it
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + 1        # word with no slot
    for i in range(1, n + 1):
        present = {w for w in slots[i - 1] if w is not None}
        for j in range(1, m + 1):
            same = 0.0 if hypothesis[j - 1] in present else 1.0
            cost[i][j] = min(cost[i - 1][j - 1] + same,
                             cost[i - 1][j] + 1,
                             cost[i][j - 1] + 1)

    out, i, j = [], n, m
    while i > 0 or j > 0:
        present = {w for w in slots[i - 1] if w is not None} if i > 0 else set()
        same = 0.0 if (i > 0 and j > 0 and hypothesis[j - 1] in present) else 1.0
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + same:
            out.append(slots[i - 1] + [hypothesis[j - 1]]); i -= 1; j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            out.append(slots[i - 1] + [None]); i -= 1
        else:
            out.append([None] * system_count + [hypothesis[j - 1]]); j -= 1
    out.reverse()
    return out


def combine(hypotheses, weights):
    """hypotheses: list of (system name, word list), best system first."""
    names = [name for name, _ in hypotheses]
    slots = [[w] for w in hypotheses[0][1]]
    for index, (_, hypothesis) in enumerate(hypotheses[1:], start=1):
        slots = align_into(slots, hypothesis, index)

    picked = []
    for slot in slots:
        score = {}
        for name, word in zip(names, slot):
            key = word if word is not None else None
            bucket = 'other' if word is None else script_of(word)
            score[key] = score.get(key, 0.0) + weights[name][bucket]
        best = max(score, key=lambda k: (score[k], k is not None))
        if best is not None:
            picked.append(best)
    return picked


def wer(reference, hypothesis):
    r, h = reference, hypothesis
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)], len(r)


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if '\t' in line:
            name, text = line.split('\t', 1)
            out[name] = text
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'clean'
    base = Path('fixtures/mixed-ru-en')
    manifest = json.loads((base / 'manifest.json').read_text())
    order = ['parakeet', 'gigaam', 'fastconf']
    hyp = {name: load(f'target/dev-bundles/hyp/{which}.{name}.tsv') for name in order}

    rows = {name: [0, 0] for name in order}
    rows['ROVER (плоский)'] = [0, 0]
    rows['ROVER + алфавит'] = [0, 0]
    per_kind = {}

    for entry in manifest:
        reference = words(entry['text'])
        systems = [(name, words(hyp[name].get(entry['file'], ''))) for name in order]
        for name, h in systems:
            e, n = wer(reference, h); rows[name][0] += e; rows[name][1] += n
        for label, w in (('ROVER (плоский)', PLAIN), ('ROVER + алфавит', WEIGHTS)):
            e, n = wer(reference, combine(systems, w))
            rows[label][0] += e; rows[label][1] += n
            per_kind.setdefault(entry['kind'], {}).setdefault(label, [0, 0])
            per_kind[entry['kind']][label][0] += e
            per_kind[entry['kind']][label][1] += n
        for name, h in systems:
            per_kind.setdefault(entry['kind'], {}).setdefault(name, [0, 0])
            per_kind[entry['kind']][name][0] += wer(reference, h)[0]
            per_kind[entry['kind']][name][1] += len(reference)

    print(f'=== {which} ===')
    print(f"{'система':<18} {'WER':>7}")
    for name, (e, n) in rows.items():
        print(f'{name:<18} {e / max(1, n):>6.0%}')
    print()
    print(f"{'':<18} " + " ".join(f'{k:>7}' for k in ('ru', 'en', 'mix')))
    for name in list(rows):
        cells = []
        for kind in ('ru', 'en', 'mix'):
            e, n = per_kind.get(kind, {}).get(name, [0, 1])
            cells.append(f'{e / max(1, n):>6.0%}')
        print(f'{name:<18} ' + " ".join(f'{c:>7}' for c in cells))


if __name__ == '__main__':
    main()
