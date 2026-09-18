//! Add a repair to the dictionary from the command line.
//!
//! The counterpart of the button in the dictionary panel, for seeding a term
//! the user already knows the app gets wrong. It goes through `Store::teach`
//! rather than SQL so the key is folded exactly the way matching folds it.
//!
//!     cargo run -p shi-store --example teach -- <db> <heard> <should be>
//!
//! With no arguments after the database it lists what is already there, and
//! `--on "<line>"` shows what the dictionary would do to that line.

use shi_store::Store;

fn main() {
    let all: Vec<String> = std::env::args().skip(1).collect();

    // `--on <line>` is pulled out first so it cannot be mistaken for one of
    // the two positional words.
    let sample = all
        .iter()
        .position(|a| a == "--on")
        .and_then(|at| all.get(at + 1).cloned());
    let mut positional = all.iter();
    let mut words: Vec<String> = Vec::new();
    while let Some(argument) = positional.next() {
        if argument == "--on" {
            positional.next();
            continue;
        }
        words.push(argument.clone());
    }

    let database = words.first().expect("usage: teach <db> [<heard> <should be>] [--on <line>]");
    let store = Store::open(database).expect("open the meetings database");

    if let (Some(heard), Some(correct)) = (words.get(1), words.get(2)) {
        match store.teach(heard, correct, &jiff::Zoned::now().to_string()) {
            Ok(()) => println!("выучено: {heard} → {correct}"),
            Err(error) => {
                eprintln!("не записано: {error}");
                std::process::exit(1);
            }
        }
    }

    if let Some(line) = sample {
        let (repaired, fired) = store.dictionary().expect("compile").apply(&line);
        println!("  было:  {line}");
        println!("  стало: {repaired}");
        println!("  сработало правил: {}", fired.len());
    }

    for rule in store.corrections().expect("read the dictionary") {
        println!(
            "  {} → {}  (ключ {}, применено {})",
            rule.heard, rule.right, rule.wrong, rule.hits
        );
    }
}
