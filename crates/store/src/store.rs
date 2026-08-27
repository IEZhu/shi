use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use shi_audio::StreamKind;

use crate::error::{Result, StoreError};
use crate::model::{
    MATCH_CLOSE, MATCH_OPEN, Meeting, NewSegment, Segment, SearchHit, SessionSlot, Speaker,
    SpeakerVoice, blob_to_embedding, embedding_to_blob,
};
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
            "INSERT INTO segments
                (meeting_id, stream, t_start_ms, t_end_ms, speaker_id, session_slot, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                meeting_id,
                segment.stream.as_str(),
                segment.start_ms,
                segment.end_ms,
                segment.speaker_id,
                segment.session_slot,
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
            "SELECT s.id, s.stream, s.t_start_ms, s.t_end_ms,
                    s.speaker_id, p.display_name, s.session_slot, s.text
             FROM segments s
             LEFT JOIN speakers p ON p.id = s.speaker_id
             WHERE s.meeting_id = ?1
             ORDER BY s.t_start_ms, s.id",
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
                speaker_id: row.get(4)?,
                speaker_name: row.get(5)?,
                session_slot: row.get(6)?,
                text: row.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    // ---- speakers -------------------------------------------------------

    /// Find or create a person by name.
    pub fn speaker_named(&self, display_name: &str, now: &str) -> Result<Speaker> {
        self.connection.execute(
            "INSERT OR IGNORE INTO speakers (display_name, created_at) VALUES (?1, ?2)",
            params![display_name, now],
        )?;
        Ok(self.connection.query_row(
            "SELECT id, display_name, created_at, notes FROM speakers WHERE display_name = ?1",
            params![display_name],
            speaker_from_row,
        )?)
    }

    pub fn speakers(&self) -> Result<Vec<Speaker>> {
        let mut statement = self.connection.prepare(
            "SELECT id, display_name, created_at, notes FROM speakers ORDER BY display_name",
        )?;
        let rows = statement.query_map([], speaker_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Rename someone.
    ///
    /// One row, and every transcript they ever appeared in is correct the next
    /// time it renders. This is what storing a reference rather than a copied
    /// name buys.
    pub fn rename_speaker(&self, speaker_id: i64, display_name: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE speakers SET display_name = ?2 WHERE id = ?1",
            params![speaker_id, display_name],
        )?;
        Ok(())
    }

    /// Remove a person and every voiceprint of them.
    ///
    /// Segments that pointed at them fall back to their session slot, so the
    /// transcript reverts to "Спикер N" rather than losing the attribution
    /// altogether — the app still knows those lines were one voice.
    pub fn forget_speaker(&self, speaker_id: i64) -> Result<()> {
        // ON DELETE CASCADE clears voiceprints; the segment and slot columns
        // are ON DELETE SET NULL.
        self.connection.execute(
            "DELETE FROM speakers WHERE id = ?1",
            params![speaker_id],
        )?;
        Ok(())
    }

    // ---- voiceprints ----------------------------------------------------

    pub fn add_voiceprint(
        &self,
        speaker_id: i64,
        embedding: &[f32],
        model_id: &str,
        source_meeting_id: Option<i64>,
        duration_ms: i64,
        now: &str,
    ) -> Result<i64> {
        self.connection.execute(
            "INSERT INTO voiceprints
                (speaker_id, embedding, dim, model_id, source_meeting_id, duration_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                speaker_id,
                embedding_to_blob(embedding),
                embedding.len() as i64,
                model_id,
                source_meeting_id,
                duration_ms,
                now,
            ],
        )?;
        Ok(self.connection.last_insert_rowid())
    }

    /// Every named voice recorded with `model_id`, ready for matching.
    ///
    /// Filtered by model because embeddings from different models are not
    /// comparable: cosine similarity between them is noise, not evidence.
    pub fn voices_for_model(&self, model_id: &str) -> Result<Vec<SpeakerVoice>> {
        let mut statement = self.connection.prepare(
            "SELECT s.id, s.display_name, v.embedding
             FROM speakers s
             JOIN voiceprints v ON v.speaker_id = s.id
             WHERE v.model_id = ?1
             ORDER BY s.id",
        )?;

        let mut voices: Vec<SpeakerVoice> = Vec::new();
        let rows = statement.query_map(params![model_id], |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            Ok((id, name, blob_to_embedding(&blob)))
        })?;

        for row in rows {
            let (id, name, embedding) = row?;
            match voices.last_mut() {
                Some(voice) if voice.speaker_id == id => voice.embeddings.push(embedding),
                _ => voices.push(SpeakerVoice {
                    speaker_id: id,
                    display_name: name,
                    embeddings: vec![embedding],
                }),
            }
        }
        Ok(voices)
    }

    // ---- session slots --------------------------------------------------

    /// Record or update an unnamed voice heard in this meeting.
    pub fn upsert_session_slot(&self, slot: &SessionSlot) -> Result<()> {
        self.connection.execute(
            "INSERT INTO session_slots
                (meeting_id, slot, centroid, dim, model_id, sample_path,
                 total_speech_ms, utterances)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT (meeting_id, slot) DO UPDATE SET
                centroid = excluded.centroid,
                total_speech_ms = excluded.total_speech_ms,
                utterances = excluded.utterances,
                sample_path = COALESCE(excluded.sample_path, session_slots.sample_path)",
            params![
                slot.meeting_id,
                slot.slot,
                embedding_to_blob(&slot.centroid),
                slot.centroid.len() as i64,
                slot.model_id,
                slot.sample_path,
                slot.total_speech_ms,
                slot.utterances,
            ],
        )?;
        Ok(())
    }

    pub fn session_slots(&self, meeting_id: i64) -> Result<Vec<SessionSlot>> {
        let mut statement = self.connection.prepare(
            "SELECT meeting_id, slot, centroid, model_id, sample_path,
                    total_speech_ms, utterances, resolved_speaker_id
             FROM session_slots WHERE meeting_id = ?1 ORDER BY slot",
        )?;
        let rows = statement.query_map(params![meeting_id], |row| {
            let blob: Vec<u8> = row.get(2)?;
            Ok(SessionSlot {
                meeting_id: row.get(0)?,
                slot: row.get(1)?,
                centroid: blob_to_embedding(&blob),
                model_id: row.get(3)?,
                sample_path: row.get(4)?,
                total_speech_ms: row.get(5)?,
                utterances: row.get(6)?,
                resolved_speaker_id: row.get(7)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    // ---- search ---------------------------------------------------------

    /// Find transcript lines matching what the user typed.
    ///
    /// Ordered by relevance rather than date: someone searching for a decision
    /// wants the line where it was made, not the most recent meeting.
    pub fn search(&self, input: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let Some(query) = fts_query(input) else {
            return Ok(Vec::new());
        };

        let mut statement = self.connection.prepare(
            "SELECT m.id, m.title, m.started_at, s.id, s.t_start_ms,
                    p.display_name, s.session_slot,
                    snippet(segments_fts, 0, ?2, ?3, '…', 14)
             FROM segments_fts
             JOIN segments s ON s.id = segments_fts.rowid
             JOIN meetings m ON m.id = s.meeting_id
             LEFT JOIN speakers p ON p.id = s.speaker_id
             WHERE segments_fts MATCH ?1
             ORDER BY rank
             LIMIT ?4",
        )?;

        let rows = statement.query_map(
            params![
                query,
                MATCH_OPEN.to_string(),
                MATCH_CLOSE.to_string(),
                limit as i64
            ],
            |row| {
                Ok(SearchHit {
                    meeting_id: row.get(0)?,
                    meeting_title: row.get(1)?,
                    meeting_started_at: row.get(2)?,
                    segment_id: row.get(3)?,
                    start_ms: row.get(4)?,
                    speaker_name: row.get(5)?,
                    session_slot: row.get(6)?,
                    snippet: row.get(7)?,
                })
            },
        )?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Meetings that started before `cutoff`, for pruning their audio.
    ///
    /// Compares the stored timestamp as a string, which works because it is
    /// always written in a fixed-width, zone-annotated format that sorts
    /// chronologically for a given zone. The cutoff is produced the same way.
    pub fn meetings_started_before(&self, cutoff: &str) -> Result<Vec<i64>> {
        let mut statement = self
            .connection
            .prepare("SELECT id, started_at FROM meetings")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        let cutoff_instant = instant_of(cutoff);
        Ok(rows
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, started)| instant_of(started) < cutoff_instant)
            .map(|(id, _)| id)
            .collect())
    }

    /// Remove a meeting and everything recorded with it.
    pub fn delete_meeting(&self, meeting_id: i64) -> Result<()> {
        self.connection
            .execute("DELETE FROM meetings WHERE id = ?1", params![meeting_id])?;
        Ok(())
    }

    /// Give an unnamed voice a name.
    ///
    /// Claims every segment of that slot, stores the slot centroid as a
    /// voiceprint so the person is recognised in future meetings, and marks
    /// the slot resolved — all in one transaction, because a half-applied
    /// naming would leave the transcript disagreeing with itself.
    pub fn name_session_slot(
        &mut self,
        meeting_id: i64,
        slot: u32,
        display_name: &str,
        now: &str,
    ) -> Result<Speaker> {
        let transaction = self.connection.transaction()?;

        transaction.execute(
            "INSERT OR IGNORE INTO speakers (display_name, created_at) VALUES (?1, ?2)",
            params![display_name, now],
        )?;
        let speaker: Speaker = transaction.query_row(
            "SELECT id, display_name, created_at, notes FROM speakers WHERE display_name = ?1",
            params![display_name],
            speaker_from_row,
        )?;

        let stored: Option<(Vec<u8>, String, i64)> = transaction
            .query_row(
                "SELECT centroid, model_id, total_speech_ms
                 FROM session_slots WHERE meeting_id = ?1 AND slot = ?2",
                params![meeting_id, slot],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        if let Some((centroid, model_id, speech_ms)) = stored {
            transaction.execute(
                "INSERT INTO voiceprints
                    (speaker_id, embedding, dim, model_id, source_meeting_id,
                     duration_ms, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    speaker.id,
                    &centroid,
                    (centroid.len() / 4) as i64,
                    model_id,
                    meeting_id,
                    speech_ms,
                    now,
                ],
            )?;
        }

        transaction.execute(
            "UPDATE session_slots SET resolved_speaker_id = ?3
             WHERE meeting_id = ?1 AND slot = ?2",
            params![meeting_id, slot, speaker.id],
        )?;
        transaction.execute(
            "UPDATE segments SET speaker_id = ?3
             WHERE meeting_id = ?1 AND session_slot = ?2",
            params![meeting_id, slot, speaker.id],
        )?;

        transaction.commit()?;
        Ok(speaker)
    }
}

/// Compare timestamps as instants rather than text.
///
/// Two meetings in different time zones sort wrongly as strings, and a laptop
/// that travels produces exactly that. Anything unparseable sorts as the epoch,
/// so a corrupt row is pruned rather than kept forever.
fn instant_of(value: &str) -> i128 {
    value
        .parse::<jiff::Zoned>()
        .map(|z| z.timestamp().as_nanosecond())
        .or_else(|_| value.parse::<jiff::Timestamp>().map(|t| t.as_nanosecond()))
        .unwrap_or(i128::MIN)
}

/// Turn what a person typed into an FTS5 query.
///
/// User input reaches FTS5's own query language, where a stray quote or the
/// word "AND" is a syntax error rather than a search. Quoting every term
/// removes that surface entirely, and the last term gets a prefix wildcard so
/// results appear while they are still typing.
pub fn fts_query(input: &str) -> Option<String> {
    let terms: Vec<String> = input
        .split_whitespace()
        .map(|term| term.replace('"', ""))
        .filter(|term| !term.is_empty())
        .collect();

    if terms.is_empty() {
        return None;
    }

    let last = terms.len() - 1;
    Some(
        terms
            .iter()
            .enumerate()
            .map(|(index, term)| {
                if index == last {
                    format!("\"{term}\"*")
                } else {
                    format!("\"{term}\"")
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn speaker_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Speaker> {
    Ok(Speaker {
        id: row.get(0)?,
        display_name: row.get(1)?,
        created_at: row.get(2)?,
        notes: row.get(3)?,
    })
}
