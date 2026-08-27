//! Install a model from the command line.
//!
//! The app does this itself; this exists for setting up a checkout, and for
//! repairing an installation without launching anything.
//!
//!     cargo run -p shi-models --example install -- <models-dir> <model-id>...

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: install <models-dir> <model-id>..."));
    let ids: Vec<String> = args.collect();

    let ids = if ids.is_empty() {
        shi_models::required(None)
            .into_iter()
            .map(|spec| spec.id.to_string())
            .collect()
    } else {
        ids
    };

    for id in ids {
        let Some(spec) = shi_models::by_id(&id) else {
            eprintln!("unknown model: {id}");
            continue;
        };
        if spec.installed(&dir) {
            println!("{:<28} already installed", spec.id);
            continue;
        }

        let started = Instant::now();
        let mut last = u64::MAX;
        let result = shi_models::install(
            spec,
            &dir,
            |progress| {
                let percent = progress.downloaded * 100 / progress.total.max(1);
                if percent != last && percent % 10 == 0 {
                    last = percent;
                    print!("\r{:<28} {percent:>3}%", spec.id);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            },
            || true,
        );

        match result {
            Ok(path) => println!(
                "\r{:<28} installed in {:.0}s -> {}",
                spec.id,
                started.elapsed().as_secs_f32(),
                path.display()
            ),
            Err(err) => println!("\r{:<28} FAILED: {err}", spec.id),
        }
    }
}
