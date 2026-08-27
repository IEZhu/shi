//! What is installed, and what a meeting would still be missing.
//!
//!     cargo run -p shi-models --example status -- <models-dir> [recognizer-id]

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: status <models-dir> [recognizer-id]"));
    let chosen = args.next();

    println!("{}\n", dir.display());
    for spec in shi_models::CATALOGUE {
        let mark = if spec.installed(&dir) { "installed" } else { "-" };
        println!("  {:<26} {:<10} {}", spec.id, mark, spec.summary);
    }

    let missing: Vec<&str> = shi_models::required(chosen.as_deref())
        .into_iter()
        .filter(|spec| !spec.installed(&dir))
        .map(|spec| spec.display_name)
        .collect();

    println!();
    if missing.is_empty() {
        println!("ready: a meeting can start");
    } else {
        println!("missing: {}", missing.join(", "));
    }
}
