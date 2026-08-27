use std::path::Path;

use rusqlite::{Connection, params};
use shi_audio::StreamKind;

use crate::error::{Result, StoreError};
use crate::model::{Meeting, NewSegment, Segment};
use crate::schema;

/// The transcript database.
///
/// This is the source of truth for a meeting; the Markdown file is a
/// projection rendered from it. That ordering is what lets a speaker named
/// after the fact fix every line they ever said.
pub struct Store {
    connection: Connection,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path)?;
        schema::migrate(&connection)?;
        Ok(Self { connection })
    }

    /// In-memory database, for tests.
    pub fn in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()?;
        schema::migrate(&connection)?;
        Ok(Self { connection })
    }

    pub fn start_meeting(
        &self,
        title: &str,
        started_at: &str,
        stt_model_id: &str,
    ) -> Result<Meeting> {
        self.connection.execute(
            "INSERT INTO meetings (title, started_at, stt_model_id) VALUES (?1, ?2, ?3)",
            params![title, started_at, stt_model_id],
        )?;
        self.meeting(self.connection.last_insert_rowid())
    }

    pub fn finish_meeting(&self, meeting_id: i64, ended_at: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE meetings SET ended_at = ?2 WHERE id = ?1",
            params![meeting_id, ended_at],
        )?;
        Ok(())
    }

    pub fn set_markdown_path(&self, meeting_id: i64, path: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE meetings SET md_path = ?2 WHERE id = ?1",
            params![meeting_id, path],
        )?;
        Ok(())
    }

    pub fn meeting(&self, id: i64) -> Result<Meeting> {
        self.connection
            .query_row(
                "SELECT id, title, started_at, ended_at, stt_model_id, md_path, audio_dir
                 FROM meetings WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Meeting {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        started_at: row.get(2)?,
                        ended_at: row.get(3)?,
                        stt_model_id: row.get(4)?,
                        md_path: row.get(5)?,
                        audio_dir: row.get(6)?,
                    })
                },
            )
            .map_err(|err| match err {
                rusqlite::Error::QueryReturnedNoRows => StoreError::UnknownMeeting(id),
                other => other.into(),
            })
    }

    pub fn meetings(&self) -> Result<Vec<Meeting>> {
        let mut statement = self.connection.prepare(
            "SELECT id, title, started_at, ended_at, stt_model_id, md_path, audio_dir
             FROM meetings ORDER BY started_at DESC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(Meeting {
                id: row.get(0)?,
                title: row.get(1)?,
                started_at: row.get(2)?,
                ended_at: row.get(3)?,
                stt_model_id: row.get(4)?,
                md_path: row.get(5)?,
                audio_dir: row.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Record a finalised utterance. Called as each one closes, so a crash
    /// costs at most the utterance still being spoken.
    pub fn append_segment(&self, meeting_id: i64, segment: &NewSegment) -> Result<i64> {
        self.connection.execute(
            "INSERT INTO segments (meeting_id, stream, t_start_ms, t_end_ms, speaker, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                meeting_id,
                segment.stream.as_str(),
                segment.start_ms,
                segment.end_ms,
                segment.speaker,
                segment.text,
            ],
        )?;
        Ok(self.connection.last_insert_rowid())
    }

    /// Every segment of a meeting, in the order it was spoken.
    ///
    /// The two streams interleave here: this is where a microphone utterance
    /// and a system-audio utterance become one conversation.
    pub fn segments(&self, meeting_id: i64) -> Result<Vec<Segment>> {
        let mut statement = self.connection.prepare(
            "SELECT id, stream, t_start_ms, t_end_ms, speaker, text
             FROM segments WHERE meeting_id = ?1
             ORDER BY t_start_ms, id",
        )?;
        let rows = statement.query_map(params![meeting_id], |row| {
            let stream: String = row.get(1)?;
            Ok(Segment {
                id: row.get(0)?,
                stream: if stream == "mic" {
                    StreamKind::Mic
                } else {
                    StreamKind::System
                },
                start_ms: row.get(2)?,
                end_ms: row.get(3)?,
                speaker: row.get(4)?,
                text: row.get(5)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Give every segment currently attributed to `from` the name `to`.
    ///
    /// This is the whole point of keeping Markdown as a projection: naming a
    /// voice after the meeting fixes the entire transcript in one statement.
    /// Passing `None` for `from` claims the not-yet-identified segments.
    pub fn rename_speaker(
        &self,
        meeting_id: i64,
        from: Option<&str>,
        to: &str,
    ) -> Result<usize> {
        let changed = match from {
            Some(from) => self.connection.execute(
                "UPDATE segments SET speaker = ?3 WHERE meeting_id = ?1 AND speaker = ?2",
                params![meeting_id, from, to],
            )?,
            None => self.connection.execute(
                "UPDATE segments SET speaker = ?2 WHERE meeting_id = ?1 AND speaker IS NULL",
                params![meeting_id, to],
            )?,
        };
        Ok(changed)
    }
}
