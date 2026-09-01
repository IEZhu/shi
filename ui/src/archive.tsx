import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  type ArchivedLine,
  type EditOutcome,
  type MeetingSummary,
  type Reprocessed,
  type SearchResult,
  formatMeetingStart,
  formatOffset,
  speakerLabel,
  splitSnippet,
} from "./types";

/** Long enough that a fast typist issues one query, not eight. */
const DEBOUNCE_MS = 180;

function Snippet({ snippet }: { snippet: string }) {
  return (
    <>
      {splitSnippet(snippet).map((part, index) =>
        part.match ? (
          <mark key={index}>{part.text}</mark>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  );
}

/**
 * Past meetings, and search across all of them.
 *
 * Search results come back ranked by relevance rather than date: someone
 * looking for a decision wants the line where it was made, not the most recent
 * meeting to mention the word.
 */
export function Archive({ onClose }: { onClose: () => void }) {
  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [open, setOpen] = useState<MeetingSummary | null>(null);
  const [lines, setLines] = useState<ArchivedLine[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    setMeetings(await invoke<MeetingSummary[]>("meetings"));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    if (query.trim() === "") {
      setResults(null);
      return;
    }
    timer.current = window.setTimeout(async () => {
      setResults(await invoke<SearchResult[]>("search", { query }));
    }, DEBOUNCE_MS);
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, [query]);

  const openMeeting = async (meeting: MeetingSummary) => {
    setOpen(meeting);
    setNote(null);
    setEditing(null);
    setLines(await invoke<ArchivedLine[]>("meeting_transcript", { meetingId: meeting.id }));
  };

  /**
   * Save an edited line, and keep whatever it taught about the vocabulary.
   *
   * The whole transcript is reloaded rather than the one line patched in
   * place, because a rule learned here repairs every other line that says the
   * same word — which is the entire reason to edit rather than fix the
   * Markdown by hand.
   */
  const saveEdit = async (meeting: MeetingSummary, line: ArchivedLine) => {
    const text = draft.trim();
    setEditing(null);
    if (text === "" || text === line.text) return;

    const outcome = await invoke<EditOutcome>("edit_segment", {
      meetingId: meeting.id,
      segmentId: line.id,
      text,
    });
    setLines(await invoke<ArchivedLine[]>("meeting_transcript", { meetingId: meeting.id }));
    setNote(
      outcome.learned === 0
        ? "Реплика исправлена."
        : `Реплика исправлена, и запомнено правил: ${outcome.learned}. ` +
          "Это слово теперь чинится во всех расшифровках.",
    );
  };

  const startEdit = (line: ArchivedLine) => {
    setEditing(line.id);
    setDraft(line.text);
  };

  /// Re-run attribution over the recording, which sees the whole meeting at
  /// once rather than deciding as it goes.
  const reprocess = async (meeting: MeetingSummary) => {
    setBusy(true);
    setNote(null);
    try {
      const result = await invoke<Reprocessed>("reprocess_speakers", {
        meetingId: meeting.id,
      });
      setLines(await invoke<ArchivedLine[]>("meeting_transcript", { meetingId: meeting.id }));
      setNote(
        result.speakersBefore === result.speakersAfter && result.renamed === 0
          ? `Ничего не изменилось: те же ${result.speakersAfter} голоса.`
          : `Голосов было ${result.speakersBefore}, стало ${result.speakersAfter}; ` +
            `переназначено реплик: ${result.renamed}.`,
      );
    } catch (error) {
      setNote(String(error));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (meeting: MeetingSummary) => {
    await invoke("delete_meeting", { meetingId: meeting.id });
    if (open?.id === meeting.id) setOpen(null);
    refresh();
  };

  const body = useMemo(() => {
    if (open) {
      return (
        <div className="transcript archive-transcript">
          {lines.map((line) => (
            <p key={line.id} className={`turn stream-${line.stream}`}>
              <span className="turn-head">
                <span className="time">{formatOffset(line.startMs)}</span>
                <span className="who">
                  {speakerLabel(line.stream, line.speaker, line.slot)}
                </span>
              </span>
              {editing === line.id ? (
                <textarea
                  className="said editable"
                  value={draft}
                  autoFocus
                  rows={Math.max(1, Math.ceil(draft.length / 72))}
                  onChange={(event) => setDraft(event.target.value)}
                  onBlur={() => saveEdit(open, line)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      saveEdit(open, line);
                    }
                    if (event.key === "Escape") setEditing(null);
                  }}
                />
              ) : (
                <span
                  className="said correctable"
                  title="Исправить — слово запомнится и починится везде"
                  onClick={() => startEdit(line)}
                >
                  {line.text}
                </span>
              )}
            </p>
          ))}
          {lines.length === 0 && <p className="subtitle">Транскрипт пуст.</p>}
        </div>
      );
    }

    if (results !== null) {
      if (results.length === 0) {
        return <p className="subtitle">Ничего не нашлось.</p>;
      }
      return (
        <ul className="hits">
          {results.map((hit, index) => (
            <li key={`${hit.meetingId}-${hit.startMs}-${index}`}>
              <div className="voice-head">
                <strong>{hit.meetingTitle}</strong>
                <span className="voice-meta">
                  {formatMeetingStart(hit.meetingStartedAt)} · {formatOffset(hit.startMs)} ·{" "}
                  {speakerLabel("system", hit.speaker, hit.slot)}
                </span>
              </div>
              <p className="said">
                <Snippet snippet={hit.snippet} />
              </p>
            </li>
          ))}
        </ul>
      );
    }

    if (meetings.length === 0) {
      return <p className="subtitle">Пока ни одной встречи.</p>;
    }
    return (
      <ul className="voices">
        {meetings.map((meeting) => (
          <li key={meeting.id}>
            <div className="voice-head">
              <strong>{meeting.title}</strong>
              <span className="voice-meta">
                {formatMeetingStart(meeting.startedAt)} · {meeting.segmentCount} реплик
              </span>
            </div>
            <div className="toast-row">
              <button className="ghost" onClick={() => openMeeting(meeting)}>
                Открыть
              </button>
              <button className="ghost" onClick={() => remove(meeting)}>
                Удалить
              </button>
              {meeting.mdPath && <span className="voice-meta path">{meeting.mdPath}</span>}
            </div>
          </li>
        ))}
      </ul>
    );
  }, [open, lines, results, meetings, editing, draft]);

  return (
    <section className="review archive">
      <header>
        <div>
          <h2>{open ? open.title : "Архив встреч"}</h2>
          <p className="subtitle">
            {open
              ? formatMeetingStart(open.startedAt)
              : "Поиск идёт по всем расшифровкам сразу."}
          </p>
        </div>
        <div className="toast-row">
          {!open && (
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Найти в расшифровках"
              aria-label="Поиск по расшифровкам"
            />
          )}
          {open && (
            <button className="ghost" onClick={() => reprocess(open)} disabled={busy}>
              Пересобрать говорящих
            </button>
          )}
          <button className="ghost" onClick={() => (open ? setOpen(null) : onClose())}>
            {open ? "Назад" : "Закрыть"}
          </button>
        </div>
      </header>
      {note && <p className="notice">{note}</p>}
      {body}
    </section>
  );
}
