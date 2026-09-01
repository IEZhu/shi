//! Does a correction taught on one recording repair a different one?
//!
//! The dictionary's whole claim is that fixing a term once fixes it later. That
//! only holds if the recogniser mangles the term the same way twice, and
//! `docs/transcription.md` records that it does not always — the same phrase
//! came back as "сред пул" on clean audio and "средпул" through a telephone
//! band. This measures what survives the difference.
//!
//! Teaches from the clean corpus, applies to the degraded one, and reports the
//! word error rate on the mixed-language sentences, which is where every
//! remaining error lives.
//!
//!     cargo run -p shi-store --example dictionary_transfer -- \
//!         target/dev-bundles/hyp/clean.parakeet.tsv \
//!         target/dev-bundles/hyp/degraded.parakeet.tsv

use std::collections::HashMap;

use shi_store::corrections::{self, Corrections};

/// The reference sentences, matching `fixtures/mixed-ru-en/manifest.json`.
///
/// The monolingual sentences are here so the number is comparable with the
/// tables in `docs/transcription.md`, and so a rule that damages plain Russian
/// shows up rather than hiding behind the mixed ones.
const REFERENCE: &[(&str, &str)] = &[
    ("00-ru.wav", "Мы увеличили количество шардов и снизили нагрузку на горячие ноды."),
    ("01-ru.wav", "Максимальное сжатие на приёме стоило нам слишком дорого."),
    ("02-ru.wav", "Индексы отсортировали по объёму поступающих данных."),
    ("03-en.wav", "The broker lost its leader partition and the consumer group rebalanced."),
    ("04-en.wav", "We increased the heap size on the hot nodes before the next deployment."),
    ("05-en.wav", "Elasticsearch stores incoming documents in segments on the hot tier."),
    ("06-mix.wav", "Мы посмотрели в Kibana, там consumer lag вырос, и Elasticsearch начал отдавать 429."),
    ("07-mix.wav", "Нужно пересчитать shard allocation, иначе rebalance будет каждые пять минут."),
    ("08-mix.wav", "Kafka не могла сделать запись в consumer offsets topic по метаданным."),
    ("09-mix.wav", "Сервис отдаёт timeout, потому что thread pool забит."),
];

fn read(path: &str) -> HashMap<String, String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("cannot read {path}: {err}"))
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(name, text)| (name.to_string(), text.to_string()))
        .collect()
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace('ё', "е")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn errors(reference: &[String], hypothesis: &[String]) -> usize {
    let (n, m) = (reference.len(), hypothesis.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        d[i][0] = i;
    }
    for j in 1..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let same = reference[i - 1] == hypothesis[j - 1];
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + usize::from(!same));
        }
    }
    d[n][m]
}

fn rate(pairs: &[(Vec<String>, Vec<String>)]) -> f32 {
    let (bad, total) = pairs.iter().fold((0, 0), |(bad, total), (reference, hypothesis)| {
        (bad + errors(reference, hypothesis), total + reference.len())
    });
    bad as f32 / total.max(1) as f32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let taught_on = args.next().expect("usage: dictionary_transfer <taught.tsv> <applied.tsv>");
    let applied_to = args.next().expect("a second transcript to apply the rules to");

    let taught = read(&taught_on);
    let applied = read(&applied_to);

    // What the user would do: read each mixed line and write what was said.
    // Only the mixed lines, because those are the ones worth correcting.
    let mut rules = Vec::new();
    for (file, reference) in REFERENCE.iter().filter(|(f, _)| f.contains("mix")) {
        let Some(heard) = taught.get(*file) else { continue };
        for rule in corrections::learn(heard, reference) {
            println!("выучено: {} → {}", rule.heard, rule.right);
            rules.push((rule.wrong, rule.right));
        }
    }
    let dictionary = Corrections::new(rules);

    let mut rows: Vec<(&str, Vec<_>, Vec<_>)> =
        vec![("всего", Vec::new(), Vec::new()), ("ru", Vec::new(), Vec::new()),
             ("en", Vec::new(), Vec::new()), ("mix", Vec::new(), Vec::new())];

    for (file, reference) in REFERENCE {
        let Some(heard) = applied.get(*file) else { continue };
        let reference = words(reference);
        let before = (reference.clone(), words(heard));
        let after = (reference, words(&dictionary.apply(heard).0));
        let kind = file.split(['-', '.']).nth(1).unwrap_or("");
        for (label, b, a) in rows.iter_mut() {
            if *label == "всего" || *label == kind {
                b.push(before.clone());
                a.push(after.clone());
            }
        }
    }

    println!("\nвыучено на {taught_on}\nприменено к {applied_to}\n");
    println!("{:<8} {:>8} {:>8}", "", "до", "после");
    for (label, before, after) in &rows {
        println!(
            "{label:<8} {:>7.0}% {:>7.0}%",
            rate(before) * 100.0,
            rate(after) * 100.0
        );
    }
}
