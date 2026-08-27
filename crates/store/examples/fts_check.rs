fn main() {
    let connection = rusqlite::Connection::open_in_memory().expect("open");
    let version: String = connection
        .query_row("select sqlite_version()", [], |r| r.get(0))
        .expect("version");
    println!("sqlite {version}");

    match connection.execute_batch(
        "CREATE VIRTUAL TABLE probe USING fts5(text, tokenize='unicode61');
         INSERT INTO probe(text) VALUES ('Платёжный шлюз и миграция базы');",
    ) {
        Ok(()) => {
            // Case folding and Cyrillic are what a Russian transcript needs.
            let hits: i64 = connection
                .query_row(
                    "SELECT count(*) FROM probe WHERE probe MATCH 'ШЛЮЗ'",
                    [],
                    |r| r.get(0),
                )
                .expect("query");
            println!("fts5: available, cyrillic case-insensitive match -> {hits} hit(s)");

            let prefix: i64 = connection
                .query_row(
                    "SELECT count(*) FROM probe WHERE probe MATCH 'мигра*'",
                    [],
                    |r| r.get(0),
                )
                .expect("prefix query");
            println!("fts5: prefix search -> {prefix} hit(s)");
        }
        Err(err) => println!("fts5: NOT available ({err})"),
    }
}
