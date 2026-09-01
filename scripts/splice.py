#!/usr/bin/env python3
"""Splice two monolingual recognisers into one transcript, and measure the ceiling.

This is not ROVER. Voting asks which hypothesis is most popular; splicing asks
which recogniser is competent for the word in front of it, and takes that one.
On speech that switches language inside a sentence a vote cannot help — every
system is wrong on the same word — but a splice can, if a specialist gets the
word its own language owns.

Two numbers matter, and the second one decides whether the first is worth
building:

    splice   what a concrete rule achieves — pick by the script of the word.
    oracle   what the best conceivable rule achieves — take the reference word
             whenever any system produced it. Nothing can beat this.

If the oracle sits on top of the best single model, no selection rule exists
and the idea is finished regardless of how the rule is written.
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rover import words, script_of, align_into, wer, load


def network(hypotheses):
    """Fold every hypothesis into one word transition network.

    Returns slots, each a list holding one word (or None) per system, in the
    order the systems were given.
    """
    slots = [[w] for w in hypotheses[0][1]]
    for index, (_, hypothesis) in enumerate(hypotheses[1:], start=1):
        slots = align_into(slots, hypothesis, index)
    return slots


def splice(names, slots, owner, base):
    """Pick each slot's word from whichever system owns that word's script.

    The base system decides what the sentence looks like — which words exist
    and in what alphabet each one is written. The specialists only get to
    restate the words already claimed for their own script.
    """
    at = {name: i for i, name in enumerate(names)}
    picked = []
    for slot in slots:
        word = slot[at[base]]
        if word is None:
            continue                      # the base heard nothing here
        chosen = owner.get(script_of(word))
        if chosen is not None and slot[at[chosen]] is not None:
            word = slot[at[chosen]]
        picked.append(word)
    return picked



def splice_on_disagreement(names, slots, base, ru, en):
    """Substitute only where the two Russian-capable systems fail to agree.

    Where a multilingual model and a Russian-only model produce the same word,
    the word belongs to Russian and no specialist is needed. Where they differ,
    both were guessing at something Russian does not own — which on this corpus
    is always an English technical term written in Cyrillic. That slot, and only
    that slot, is handed to the English model.

    The ensemble is used as an uncertainty detector rather than as a chooser,
    which the oracle says is the one thing it is good for.
    """
    at = {name: i for i, name in enumerate(names)}
    picked = []
    for slot in slots:
        word = slot[at[base]]
        if word is None:
            continue
        agreed = slot[at[ru]] is not None and slot[at[ru]] == word
        english = slot[at[en]]
        if not agreed and english is not None and script_of(english) == 'lat':
            word = english
        picked.append(word)
    return picked


def oracle(reference, slots):
    """Fewest errors any per-slot selection could produce.

    A slot may emit any word it holds, or nothing at all. Emitting nothing is
    free — an ensemble is never forced to speak.
    """
    n, m = len(slots), len(reference)
    present = [{w for w in slot if w is not None} for slot in slots]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        d[0][j] = j                                     # nothing produced it
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j],                            # slot stays silent
                d[i - 1][j - 1] + (0 if reference[j - 1] in present[i - 1] else 1),
                d[i][j - 1] + 1,
            )
    return d[n][m], m


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'clean'
    systems = sys.argv[2:] or ['parakeet', 'en', 'ru']
    base = systems[0]
    owner = {'lat': 'en', 'cyr': 'ru'}
    owner = {k: v for k, v in owner.items() if v in systems}

    manifest = json.loads(Path('fixtures/mixed-ru-en/manifest.json').read_text())
    hyp = {name: load(f'target/dev-bundles/hyp/{which}.{name}.tsv') for name in systems}

    extra = ['splice on disagreement'] if {'en', 'ru'} <= set(systems) else []
    labels = systems + ['splice'] + extra + ['oracle'] + [f'oracle {base}+{s}' for s in systems[1:]]
    total = {label: [0, 0] for label in labels}
    per_kind = {}

    for entry in manifest:
        reference = words(entry['text'])
        heard = [(name, words(hyp[name].get(entry['file'], ''))) for name in systems]
        slots = network(heard)

        scored = [(name, wer(reference, h)) for name, h in heard]
        scored.append(('splice', wer(reference, splice(systems, slots, owner, base))))
        if 'en' in systems and 'ru' in systems:
            scored.append((
                'splice on disagreement',
                wer(reference, splice_on_disagreement(systems, slots, base, 'ru', 'en')),
            ))
        scored.append(('oracle', oracle(reference, slots)))
        for other in systems[1:]:
            pair = [h for h in heard if h[0] in (base, other)]
            scored.append((f'oracle {base}+{other}', oracle(reference, network(pair))))

        for label, (errors, count) in scored:
            total[label][0] += errors
            total[label][1] += count
            bucket = per_kind.setdefault(entry['kind'], {}).setdefault(label, [0, 0])
            bucket[0] += errors
            bucket[1] += count

    kinds = ('ru', 'en', 'mix')
    print(f'=== {which} ===')
    print(f"{'система':<22} {'WER':>6}  " + "  ".join(f'{k:>5}' for k in kinds))
    for label in labels:
        errors, count = total[label]
        cells = []
        for kind in kinds:
            e, n = per_kind.get(kind, {}).get(label, [0, 1])
            cells.append(f'{e / max(1, n):>5.0%}')
        print(f'{label:<22} {errors / max(1, count):>5.0%}  ' + "  ".join(cells))


if __name__ == '__main__':
    main()
