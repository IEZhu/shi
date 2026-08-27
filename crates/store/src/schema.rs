use rusqlite::Connection;

use crate::error::Result;

/// Applied in order; each step runs once and is recorded by `user_version`.
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
        -- Superseded by speaker_id in migration 2.
        speaker     TEXT,
        text        TEXT    NOT NULL
    );

    CREATE INDEX segments_by_time ON segments (meeting_id, t_start_ms);
    "#,
    // 2 — identified voices
    //
    // A segment points at a person or at an unnamed voice in this meeting, and
    // the *name* is resolved when the transcript is rendered. Storing the name
    // on the segment instead would mean renaming someone had to rewrite every
    // line they ever spoke; this way it is one row.
    r#"
    CREATE TABLE speakers (
        id           INTEGER PRIMARY KEY,
        display_name TEXT    NOT NULL UNIQUE,
        created_at   TEXT    NOT NULL,
        notes        TEXT
    );

    -- Several per person on purpose: the same voice on AirPods and in a
    -- meeting room lands in different places, and matching takes the best of
    -- them rather than an average that resembles neither.
    CREATE TABLE voiceprints (
        id                INTEGER PRIMARY KEY,
        speaker_id        INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
        embedding         BLOB    NOT NULL,
        dim               INTEGER NOT NULL,
        model_id          TEXT    NOT NULL,
        source_meeting_id INTEGER REFERENCES meetings(id) ON DELETE SET NULL,
        duration_ms       INTEGER NOT NULL,
        created_at        TEXT    NOT NULL
    );

    CREATE INDEX voiceprints_by_speaker ON voiceprints (speaker_id);

    -- An unnamed voice within one meeting, waiting to be given a name.
    CREATE TABLE session_slots (
        meeting_id          INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
        slot                INTEGER NOT NULL,
        centroid            BLOB    NOT NULL,
        dim                 INTEGER NOT NULL,
        model_id            TEXT    NOT NULL,
        sample_path         TEXT,
        total_speech_ms     INTEGER NOT NULL DEFAULT 0,
        utterances          INTEGER NOT NULL DEFAULT 0,
        resolved_speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
        PRIMARY KEY (meeting_id, slot)
    );

    ALTER TABLE segments ADD COLUMN speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL;
    ALTER TABLE segments ADD COLUMN session_slot INTEGER;
    ALTER TABLE segments DROP COLUMN speaker;
    "#,
    // 3 — full-text search over transcripts
    //
    // An external-content table so the text is stored once. unicode61 folds
    // case across Cyrillic as well as Latin, which is what a Russian
    // transcript needs; the triggers keep the index honest when a segment is
    // edited or a meeting is deleted.
    r#"
    CREATE VIRTUAL TABLE segments_fts USING fts5(
        text,
        content = 'segments',
        content_rowid = 'id',
        tokenize = 'unicode61'
    );

    CREATE TRIGGER segments_fts_insert AFTER INSERT ON segments BEGIN
        INSERT INTO segments_fts (rowid, text) VALUES (new.id, new.text);
    END;

    CREATE TRIGGER segments_fts_delete AFTER DELETE ON segments BEGIN
        INSERT INTO segments_fts (segments_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    END;

    CREATE TRIGGER segments_fts_update AFTER UPDATE OF text ON segments BEGIN
        INSERT INTO segments_fts (segments_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
        INSERT INTO segments_fts (rowid, text) VALUES (new.id, new.text);
    END;

    INSERT INTO segments_fts (rowid, text) SELECT id, text FROM segments;
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
