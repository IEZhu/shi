use rusqlite::Connection;

use crate::error::Result;

/// Applied in order; each step runs once and is recorded by `user_version`.
///
/// The speaker and voiceprint tables deliberately do not exist yet. They arrive
/// with diarization in M2, as their own migration, so the migration path is
/// exercised rather than assumed to work.
const MIGRATIONS: &[&str] = &[
    // 1 — meetings and their transcript segments
    r#"
    CREATE TABLE meetings (
        id            INTEGER PRIMARY KEY,
        title         TEXT    NOT NULL,
        started_at    TEXT    NOT NULL,
        ended_at      TEXT,
        stt_model_id  TEXT    NOT NULL,
        md_path       TEXT,
        audio_dir     TEXT
    );

    -- Append-only. A crash costs at most the utterance still open, and
    -- re-labelling a speaker later is an UPDATE here plus a re-render, never a
    -- rewrite of the Markdown by hand.
    CREATE TABLE segments (
        id          INTEGER PRIMARY KEY,
        meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
        stream      TEXT    NOT NULL CHECK (stream IN ('mic', 'system')),
        t_start_ms  INTEGER NOT NULL,
        t_end_ms    INTEGER NOT NULL,
        -- Resolved speaker name once known; NULL means "not yet identified".
        speaker     TEXT,
        text        TEXT    NOT NULL
    );

    CREATE INDEX segments_by_time ON segments (meeting_id, t_start_ms);
    "#,
];

/// Bring a database up to the current schema. Safe to call on every open.
pub fn migrate(connection: &Connection) -> Result<()> {
    // WAL keeps the metering and pipeline threads from blocking each other,
    // and survives a hard kill mid-meeting.
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    connection.pragma_update(None, "synchronous", "NORMAL")?;

    let applied: u32 =
        connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;

    for (index, migration) in MIGRATIONS.iter().enumerate().skip(applied as usize) {
        connection.execute_batch(migration)?;
        // PRAGMA does not accept bound parameters.
        connection.pragma_update(None, "user_version", index as i64 + 1)?;
    }

    Ok(())
}
