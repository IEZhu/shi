//! A dictionary of word repairs, learned from the user's own edits.
//!
//! Measurement said the remaining transcription errors are a closed set of
//! named entities — Kibana, Kafka, consumer lag, thread pool — and that every
//! recogniser mangles them the same way every time. `docs/transcription.md`
//! records the experiments; the useful half of the result is that determinism:
//! a recogniser that always writes "Тыбаны" can be repaired by being told once.
//!
//! That is why this matches exact forms rather than phonetics. An earlier
//! attempt transliterated Cyrillic and matched it against a term list, and it
//! turned "полка" into "Kafka" — after phonetic folding the two sit as close
//! together as the repairs that work. Here nothing is guessed: a form is
//! replaced only if the user replaced it themselves, at least once.

use std::collections::HashMap;

/// Longest phrase learned or matched, in words.
///
/// Corrections are vocabulary, not rewriting. Someone reformulating a whole
/// sentence is fixing that sentence, and turning it into a rule would apply
/// their phrasing to meetings it was never about.
pub const MAX_PHRASE: usize = 4;

/// One learned replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Correction {
    pub id: i64,
    /// The matching key: folded letters, word boundaries dropped.
    pub wrong: String,
    /// The same form as it appeared on screen, for reading the list.
    pub heard: String,
    /// The form the user wrote, applied verbatim.
    pub right: String,
    pub created_at: String,
    /// How many times it has been applied, so a rule nobody needs is visible.
    pub hits: i64,
}

/// Case- and ё-insensitive key for one word.
///
/// The recogniser's vocabulary contains no `ё` at all, so a user who types it
/// would otherwise write a rule that never matches.
fn fold(word: &str) -> String {
    word.to_lowercase().replace('ё', "е")
}

/// A phrase key: the folded letters, with the word boundaries dropped.
///
/// Where a recogniser puts a space is a guess, and measurement says it is a
/// guess it makes differently every time — the same term came back as "сред
/// пул" on clean audio and "средпул" on poor audio, "тайм-аут" and "таймаут"
/// in the same corpus. The letters it committed to are far steadier than the
/// gaps, so the gaps are not part of the key and one rule covers every spacing.
fn key(words: &[&str]) -> String {
    words.iter().map(|w| fold(w)).collect()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric()
}

/// Split into words and the gaps between them, keeping every character.
fn tokenize(text: &str) -> (Vec<&str>, Vec<&str>) {
    let mut words = Vec::new();
    let mut gaps = Vec::new();
    let mut rest = text;

    loop {
        let start = rest.find(is_word).unwrap_or(rest.len());
        gaps.push(&rest[..start]);
        rest = &rest[start..];
        if rest.is_empty() {
            break;
        }
        let end = rest.find(|c| !is_word(c)).unwrap_or(rest.len());
        words.push(&rest[..end]);
        rest = &rest[end..];
    }

    (words, gaps)
}

/// The compiled dictionary, ready to apply to text.
#[derive(Debug, Clone, Default)]
pub struct Corrections {
    by_phrase: HashMap<String, String>,
}

impl Corrections {
    /// Accepts a rule written either way round — as the stored key, or as the
    /// spaced form a person would type — because both fold to the same letters.
    pub fn new(entries: impl IntoIterator<Item = (String, String)>) -> Self {
        let by_phrase = entries
            .into_iter()
            .filter_map(|(wrong, right)| {
                let (words, _) = tokenize(&wrong);
                let key = key(&words);
                (!key.is_empty() && words.len() <= MAX_PHRASE).then_some((key, right))
            })
            .collect();
        Self { by_phrase }
    }

    pub fn is_empty(&self) -> bool {
        self.by_phrase.is_empty()
    }

    /// Rewrite every phrase the dictionary knows, longest match first.
    ///
    /// Returns the repaired text and which rules fired, so the caller can
    /// count usage without matching a second time.
    pub fn apply(&self, text: &str) -> (String, Vec<String>) {
        if self.by_phrase.is_empty() {
            return (text.to_string(), Vec::new());
        }

        let (words, gaps) = tokenize(text);
        let mut out = String::with_capacity(text.len());
        let mut fired = Vec::new();
        let mut at = 0;

        while at < words.len() {
            out.push_str(gaps[at]);

            // A phrase may only span spaces and hyphens — the two things a
            // recogniser writes when it is unsure where a word ends. A comma
            // or a full stop is a claim about the sentence, not a guess about
            // a boundary, and two words either side of one are not one term.
            let reach = (1..=MAX_PHRASE.min(words.len() - at))
                .take_while(|n| {
                    *n == 1 || gaps[at + n - 1].chars().all(|c| c == ' ' || c == '-')
                })
                .collect::<Vec<_>>();

            let matched = reach.into_iter().rev().find_map(|n| {
                let phrase = key(&words[at..at + n]);
                self.by_phrase.get(&phrase).map(|right| (n, phrase, right))
            });

            match matched {
                Some((n, phrase, right)) => {
                    out.push_str(right);
                    fired.push(phrase);
                    at += n;
                }
                None => {
                    out.push_str(words[at]);
                    at += 1;
                }
            }
        }
        out.push_str(gaps[words.len()]);

        (out, fired)
    }
}

/// One replacement read out of an edit: the matching key, the form as it was
/// written, and the form the user wants instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Learned {
    pub wrong: String,
    pub heard: String,
    pub right: String,
}

/// Read the replacements out of one edit.
///
/// Aligns the two versions word by word and keeps each run that changed. A run
/// where either side is empty is an insertion or a deletion — the user adding a
/// missing word or dropping a filler — which is true of that line and of
/// nothing else, so it is not learned.
pub fn learn(before: &str, after: &str) -> Vec<Learned> {
    let (old, _) = tokenize(before);
    let (new, _) = tokenize(after);

    let mut learned = Vec::new();
    for (from, to) in changed_runs(&old, &new) {
        if from.is_empty() || to.is_empty() {
            continue;
        }
        if from.len() > MAX_PHRASE || to.len() > MAX_PHRASE {
            continue;
        }
        let wrong = key(&from);
        let right = to.join(" ");
        if wrong == key(&to) {
            continue; // the same letters, differently spaced or capitalised
        }
        learned.push(Learned { wrong, heard: from.join(" "), right });
    }
    learned
}

/// Word-level alignment, returning only the stretches that differ.
fn changed_runs<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<(Vec<&'a str>, Vec<&'a str>)> {
    let (n, m) = (old.len(), new.len());
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        cost[i][0] = i;
    }
    for j in 1..=m {
        cost[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let same = fold(old[i - 1]) == fold(new[j - 1]);
            cost[i][j] = if same {
                cost[i - 1][j - 1]
            } else {
                1 + cost[i - 1][j - 1].min(cost[i - 1][j]).min(cost[i][j - 1])
            };
        }
    }

    let mut runs: Vec<(Vec<&str>, Vec<&str>)> = Vec::new();
    let mut from = Vec::new();
    let mut to = Vec::new();
    let (mut i, mut j) = (n, m);

    let close = |from: &mut Vec<&'a str>, to: &mut Vec<&'a str>, runs: &mut Vec<_>| {
        if !from.is_empty() || !to.is_empty() {
            from.reverse();
            to.reverse();
            runs.push((std::mem::take(from), std::mem::take(to)));
        }
    };

    while i > 0 || j > 0 {
        let same = i > 0 && j > 0 && fold(old[i - 1]) == fold(new[j - 1]);
        if same && cost[i][j] == cost[i - 1][j - 1] {
            close(&mut from, &mut to, &mut runs);
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + 1 {
            from.push(old[i - 1]);
            to.push(new[j - 1]);
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            from.push(old[i - 1]);
            i -= 1;
        } else {
            to.push(new[j - 1]);
            j -= 1;
        }
    }
    close(&mut from, &mut to, &mut runs);

    runs.reverse();
    runs
}

/// Apply the dictionary to a batch of segments, reporting which rules fired.
///
/// Repairs happen on the way out rather than on the way in. The stored text
/// stays whatever the recogniser said, so a rule learned tomorrow fixes every
/// meeting already recorded — the same reason a speaker named after the fact
/// re-labels every line they spoke.
pub fn repair(segments: &mut [crate::model::Segment], dictionary: &Corrections) -> Vec<String> {
    if dictionary.is_empty() {
        return Vec::new();
    }
    let mut fired = Vec::new();
    for segment in segments {
        let (text, hits) = dictionary.apply(&segment.text);
        segment.text = text;
        fired.extend(hits);
    }
    fired
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dictionary(pairs: &[(&str, &str)]) -> Corrections {
        Corrections::new(
            pairs.iter().map(|(w, r)| (w.to_string(), r.to_string())),
        )
    }

    fn rule(wrong: &str, heard: &str, right: &str) -> Learned {
        Learned { wrong: wrong.into(), heard: heard.into(), right: right.into() }
    }

    #[test]
    fn one_edited_word_becomes_one_rule() {
        let learned = learn(
            "Мы посмотрели в Тыбаны, там всё видно.",
            "Мы посмотрели в Kibana, там всё видно.",
        );
        assert_eq!(learned, vec![rule("тыбаны", "Тыбаны", "Kibana")]);
    }

    #[test]
    fn a_term_spanning_two_words_is_learned_whole() {
        let learned = learn("там консумер Лэк вырос", "там consumer lag вырос");
        assert_eq!(learned, vec![rule("консумерлэк", "консумер Лэк", "consumer lag")]);
    }

    #[test]
    fn one_rule_covers_every_way_the_recogniser_splits_the_word() {
        // Measured on the fixtures: the same term came back as "сред пул" on
        // clean audio, "средпул" through a telephone band, and "тайм-аут"
        // beside "таймаут" in the same corpus. Where the boundary falls is a
        // guess; the letters are not.
        let learned = learn("потому что сред пул забит", "потому что thread pool забит");
        let corrections = Corrections::new(
            learned.iter().map(|l| (l.wrong.clone(), l.right.clone())),
        );

        for said in ["сред пул забит", "средпул забит", "сред-пул забит"] {
            let (text, fired) = corrections.apply(said);
            assert_eq!(text, "thread pool забит", "on {said:?}");
            assert_eq!(fired.len(), 1, "on {said:?}");
        }
    }

    #[test]
    fn adding_or_dropping_a_word_teaches_nothing() {
        // The user filling in a word the recogniser missed is true of that
        // line only; as a rule it would delete or insert words elsewhere.
        assert!(learn("сервис отдает таймаут", "сервис отдает таймаут всегда").is_empty());
        assert!(learn("ну вот сервис упал", "вот сервис упал").is_empty());
    }

    #[test]
    fn rewriting_a_sentence_is_not_vocabulary() {
        let learned = learn(
            "мы взяли и переделали там всё целиком заново",
            "команда переписала этот модуль с нуля за неделю",
        );
        assert!(learned.is_empty(), "learned {learned:?}");
    }

    #[test]
    fn two_separate_repairs_in_one_line() {
        let learned = learn(
            "в Тыбаны видно, что Квка молчит",
            "в Kibana видно, что Kafka молчит",
        );
        assert_eq!(
            learned,
            vec![rule("тыбаны", "Тыбаны", "Kibana"), rule("квка", "Квка", "Kafka")]
        );
    }

    #[test]
    fn a_rule_fires_whatever_the_capitalisation() {
        let corrections = dictionary(&[("тыбаны", "Kibana")]);
        let (text, fired) = corrections.apply("Тыбаны показали рост, тыбаны не врут.");
        assert_eq!(text, "Kibana показали рост, Kibana не врут.");
        assert_eq!(fired.len(), 2);
    }

    #[test]
    fn punctuation_and_spacing_survive_untouched() {
        let corrections = dictionary(&[("квка", "Kafka")]);
        let (text, _) = corrections.apply("  Квка,   молчит — совсем!  ");
        assert_eq!(text, "  Kafka,   молчит — совсем!  ");
    }

    #[test]
    fn a_phrase_does_not_match_across_punctuation() {
        // "консумер" ending a clause and "лэк" starting the next are not one
        // term, however much the dictionary would like them to be.
        let corrections = dictionary(&[("консумер лэк", "consumer lag")]);
        let (text, fired) = corrections.apply("это консумер, Лэк уже другой");
        assert_eq!(text, "это консумер, Лэк уже другой");
        assert!(fired.is_empty());
    }

    #[test]
    fn the_longest_phrase_wins() {
        let corrections = dictionary(&[("консумер", "consumer"), ("консумер лэк", "consumer lag")]);
        let (text, fired) = corrections.apply("там консумер лэк вырос");
        assert_eq!(text, "там consumer lag вырос");
        assert_eq!(fired, vec!["консумерлэк".to_string()]);
    }

    #[test]
    fn text_the_dictionary_knows_nothing_about_is_returned_as_it_was() {
        let corrections = dictionary(&[("тыбаны", "Kibana")]);
        let original = "Индексы отсортировали по объёму поступающих данных.";
        let (text, fired) = corrections.apply(original);
        assert_eq!(text, original);
        assert!(fired.is_empty());
    }

    #[test]
    fn a_rule_written_with_yo_still_matches_a_transcript_without_it() {
        // The recogniser's vocabulary has no ё, so it can never produce one.
        let corrections = dictionary(&[("шлёз", "шлюз")]);
        let (text, _) = corrections.apply("платежный шлез не отвечает");
        assert_eq!(text, "платежный шлюз не отвечает");
    }

    #[test]
    fn an_empty_dictionary_changes_nothing() {
        let corrections = Corrections::default();
        assert!(corrections.is_empty());
        let (text, fired) = corrections.apply("что угодно");
        assert_eq!(text, "что угодно");
        assert!(fired.is_empty());
    }

    #[test]
    fn a_batch_of_segments_is_repaired_and_the_hits_reported() {
        use shi_audio::StreamKind;

        let line = |id: i64, text: &str| crate::model::Segment {
            id,
            stream: StreamKind::System,
            start_ms: 0,
            end_ms: 1000,
            speaker_id: None,
            speaker_name: None,
            session_slot: None,
            text: text.into(),
        };

        let corrections = dictionary(&[("тыбаны", "Kibana"), ("квка", "Kafka")]);
        let mut segments = vec![
            line(1, "Смотрим в Тыбаны."),
            line(2, "Квка молчит, и Тыбаны тоже."),
            line(3, "Всё остальное работает."),
        ];

        let fired = repair(&mut segments, &corrections);

        assert_eq!(segments[0].text, "Смотрим в Kibana.");
        assert_eq!(segments[1].text, "Kafka молчит, и Kibana тоже.");
        assert_eq!(segments[2].text, "Всё остальное работает.");
        assert_eq!(fired.len(), 3);
    }
}
