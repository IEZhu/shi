use std::collections::BTreeSet;
use std::path::Path;

use jiff::tz::Offset;
use jiff::{Timestamp, Zoned};
use shi_audio::StreamKind;

use crate::error::{Result, StoreError};
use crate::model::{Meeting, Segment};

/// How timestamps are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timestamps {
    /// Time of day, matching the calendar entry the meeting came from.
    Wall,
    /// Offset from the start of the meeting.
    Relative,
}

#[derive(Debug, Clone)]
pub struct MarkdownOptions {
    pub timestamps: Timestamps,
    /// Name for microphone audio that identification has not claimed. This is
    /// the local participant, so it is known even before any diarization.
    pub unnamed_mic: String,
    /// Name for system audio that identification has not claimed.
    pub unnamed_system: String,
}

impl Default for MarkdownOptions {
    fn default() -> Self {
        Self {
            timestamps: Timestamps::Wall,
            unnamed_mic: "Вы".into(),
            unnamed_system: "Собеседник".into(),
        }
    }
}

impl MarkdownOptions {
    fn speaker_of(&self, segment: &Segment) -> String {
        segment.speaker.clone().unwrap_or_else(|| match segment.stream {
            StreamKind::Mic => self.unnamed_mic.clone(),
            StreamKind::System => self.unnamed_system.clone(),
        })
    }
}

/// Render a meeting as Markdown.
///
/// Always produced from scratch rather than appended to, so a speaker named
/// after the fact is reflected on every line they spoke.
pub fn render(meeting: &Meeting, segments: &[Segment], options: &MarkdownOptions) -> String {
    let start = parse_start(&meeting.started_at);

    let participants: BTreeSet<String> =
        segments.iter().map(|s| options.speaker_of(s)).collect();

    let last_ms = segments.iter().map(|s| s.end_ms).max().unwrap_or(0);
    let duration = format_offset(last_ms);

    let mut out = String::with_capacity(segments.len() * 96 + 256);

    out.push_str("---\n");
    out.push_str(&format!("title: {}\n", yaml_scalar(&meeting.title)));
    if let Some(start) = &start {
        out.push_str(&format!("date: {}\n", start.strftime("%Y-%m-%d")));
    }
    out.push_str(&format!("start: {}\n", meeting.started_at));
    out.push_str(&format!("duration: {duration}\n"));
    out.push_str(&format!(
        "participants: [{}]\n",
        participants
            .iter()
            .map(|p| yaml_scalar(p))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!("stt_model: {}\n", yaml_scalar(&meeting.stt_model_id)));
    out.push_str("---\n\n");

    // Consecutive turns by one speaker read as a single paragraph; a new
    // heading for every sentence would make an hour of talk unreadable.
    let mut current: Option<String> = None;
    for segment in segments {
        let speaker = options.speaker_of(segment);
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }

        if current.as_deref() == Some(speaker.as_str()) {
            out.push(' ');
            out.push_str(text);
            continue;
        }

        if current.is_some() {
            out.push_str("\n\n");
        }
        let stamp = match (options.timestamps, &start) {
            (Timestamps::Wall, Some(start)) => format_wall(start, segment.start_ms),
            _ => format_offset(segment.start_ms),
        };
        out.push_str(&format!("**[{stamp}] {speaker}:** {text}"));
        current = Some(speaker);
    }
    out.push('\n');

    out
}

/// Render and write, creating parent directories as needed.
pub fn write_to(
    path: impl AsRef<Path>,
    meeting: &Meeting,
    segments: &[Segment],
    options: &MarkdownOptions,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StoreError::Write {
            path: parent.display().to_string(),
            source,
        })?;
    }
    std::fs::write(path, render(meeting, segments, options)).map_err(|source| {
        StoreError::Write {
            path: path.display().to_string(),
            source,
        }
    })
}

/// Accept both the format we write and a plain RFC 3339 string.
///
/// jiff's `Zoned` needs an IANA annotation (`...[Europe/Moscow]`), which is what
/// the app records so daylight-saving transitions stay correct. A timestamp
/// that arrives with only a numeric offset — pasted in, or imported — still
/// carries enough to render a wall clock, so fall back to a fixed offset
/// rather than silently dropping to relative times.
fn parse_start(value: &str) -> Option<Zoned> {
    if let Ok(zoned) = value.parse::<Zoned>() {
        return Some(zoned);
    }
    let timestamp = value.parse::<Timestamp>().ok()?;
    Some(timestamp.to_zoned(rfc3339_offset(value)?.to_time_zone()))
}

/// Read the trailing `Z` or `±HH:MM` from an RFC 3339 timestamp.
fn rfc3339_offset(value: &str) -> Option<Offset> {
    if value.ends_with(['Z', 'z']) {
        return Some(Offset::UTC);
    }

    // Search only the time portion: the date's own hyphens are not offsets.
    let time_start = value.find('T').or_else(|| value.find(' '))?;
    let sign_at = value[time_start..]
        .rfind(['+', '-'])
        .map(|i| i + time_start)?;

    let sign = if value.as_bytes()[sign_at] == b'-' { -1 } else { 1 };
    let rest = &value[sign_at + 1..];
    let (hours, minutes) = match rest.split_once(':') {
        Some((h, m)) => (h, m),
        None if rest.len() == 4 => rest.split_at(2),
        None => (rest, "0"),
    };

    let seconds = hours.parse::<i32>().ok()? * 3600 + minutes.parse::<i32>().ok()? * 60;
    Offset::from_seconds(sign * seconds).ok()
}

fn format_offset(ms: i64) -> String {
    let total = ms.max(0) / 1000;
    format!("{:02}:{:02}:{:02}", total / 3600, (total / 60) % 60, total % 60)
}

fn format_wall(start: &Zoned, offset_ms: i64) -> String {
    let shifted = Timestamp::from_millisecond(start.timestamp().as_millisecond() + offset_ms.max(0))
        .map(|t| t.to_zoned(start.time_zone().clone()));
    match shifted {
        Ok(zoned) => zoned.strftime("%H:%M:%S").to_string(),
        Err(_) => format_offset(offset_ms),
    }
}

/// Quote a YAML scalar only when it would otherwise be misread.
fn yaml_scalar(value: &str) -> String {
    let needs_quotes = value.is_empty()
        || value.starts_with(['-', '?', ':', '#', '&', '*', '!', '|', '>', '\'', '"', '%', '@', '`', '[', '{'])
        || value.contains(": ")
        || value.contains(" #")
        || value.contains([',', '[', ']', '{', '}', '\n']);

    if needs_quotes {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_string()
    }
}
