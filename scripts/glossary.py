#!/usr/bin/env python3
"""Repair known terms in a transcript, by sound rather than by spelling.

The measurement that produced this: with three recognisers on a mixed corpus,
Russian scored 4 % and English 6 %, while sentences mixing the two scored 38 %.
Every remaining error was an English technical term spoken inside Russian
speech — "Kibana" arriving as "тыбаны", "Elasticsearch" as "лось" or
"лэстиксот", "Kafka" as "Кавка". A recogniser that has never been told these
words exist has no way to produce them.

So the glossary is applied afterwards, on the text, by transliterating each
Cyrillic token into Latin letters and comparing it with the known terms. No
model is involved; it is transliteration and edit distance, and a term is only
substituted when the match is close enough to be safe.
"""

import re

# Written the way a Russian speaker says them, letter by letter.
TRANSLIT = {
    'щ': 'sh', 'ш': 'sh', 'ч': 'ch', 'ц': 'ts', 'ю': 'yu', 'я': 'ya',
    'ж': 'zh', 'х': 'h', 'ё': 'yo', 'э': 'e', 'ы': 'i', 'й': 'y',
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'з': 'z',
    'и': 'i', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p',
    'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'ъ': '', 'ь': '',
}

# Letters that a recogniser confuses freely because they sound alike here.
FOLD = str.maketrans({'v': 'f', 'b': 'p', 'z': 's', 'd': 't', 'g': 'k',
                      'y': 'i', 'e': 'i', 'o': 'a', 'j': 'zh'})


def sound_of(word):
    """A crude phonetic key, reachable from either alphabet."""
    word = word.lower()
    latin = "".join(TRANSLIT.get(c, c) for c in word)
    latin = re.sub(r'[^a-z]', '', latin)
    folded = latin.translate(FOLD)
    # collapse doubled letters: "ellastic" and "elastic" sound the same
    return re.sub(r'(.)\1+', r'\1', folded)


def distance(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): d[i][0] = i
    for j in range(len(b) + 1): d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(a)][len(b)]


class Glossary:
    """Terms the meetings actually use, matched by sound."""

    def __init__(self, terms, tolerance=0.34):
        self.terms = list(terms)
        self.keys = [(sound_of(t), t) for t in self.terms]
        self.tolerance = tolerance

    def repair_word(self, word):
        key = sound_of(word)
        if not key:
            return word, None
        best, best_at = None, None
        for candidate_key, term in self.keys:
            gap = distance(key, candidate_key)
            # Allow more slack in a long word than in a short one, and never
            # let a two-letter token match a ten-letter term.
            allowed = max(1, round(self.tolerance * max(len(key), len(candidate_key))))
            if gap <= allowed and abs(len(key) - len(candidate_key)) <= allowed:
                if best is None or gap < best:
                    best, best_at = gap, term
        if best_at is None or best_at.lower() == word.lower():
            return word, None
        return best_at, (word, best_at)

    def repair(self, text):
        out, changes = [], []
        for token in re.findall(r'\w+|\W+', text, flags=re.UNICODE):
            if not token.strip() or not token[0].isalnum():
                out.append(token)
                continue
            fixed, change = self.repair_word(token)
            out.append(fixed)
            if change:
                changes.append(change)
        return "".join(out), changes


DEFAULT_TERMS = [
    "Elasticsearch", "Kibana", "Kafka", "Logstash", "consumer", "producer",
    "lag", "shard", "allocation", "rebalance", "topic", "offsets", "partition",
    "broker", "heap", "node", "index", "segment", "timeout", "thread", "pool",
    "deployment", "cluster", "replica", "snapshot", "ingest", "pipeline",
]
