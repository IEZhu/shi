/// Mirrors `capture::StreamStatus`.
export type Verdict = "stopped" | "failed" | "noFrames" | "silent" | "ok";

export interface StreamStatus {
  kind: "mic" | "system";
  verdict: Verdict;
  deviceName: string | null;
  sampleRate: number | null;
  channels: number | null;
  framesCaptured: number;
  framesDropped: number;
  /** Peak level since the previous tick, 0..1. */
  peak: number;
  error: string | null;
}

export interface Readiness {
  mic: StreamStatus;
  system: StreamStatus;
  /** Models the pipeline needs but cannot find. */
  missingModels: string[];
  recording: boolean;
  /** Decode seconds per second of audio, once transcription has run. */
  rtf: number | null;
  /** Utterances dropped because the microphone was hearing the speakers. */
  echoSuppressed: number;
  /** The meeting being recorded, or the one just finished. */
  meetingId: number | null;
}

/// Mirrors `session::TranscriptEvent`.
export type TranscriptEvent =
  | { kind: "draft"; stream: StreamName; startMs: number; text: string }
  | {
      kind: "final";
      id: number;
      stream: StreamName;
      startMs: number;
      endMs: number;
      /** Resolved name, when the voice is known. */
      speaker: string | null;
      /** Unnamed voice within this meeting. */
      slot: number | null;
      text: string;
    }
  | { kind: "draftAbandoned"; stream: StreamName }
  | { kind: "speakerDiscovered"; slot: number };

/** An unnamed voice awaiting a name. */
export interface UnnamedVoice {
  slot: number;
  totalSpeechMs: number;
  utterances: number;
  excerpt: string | null;
}

export type StreamName = "mic" | "system";

export interface Turn {
  key: string;
  stream: StreamName;
  startMs: number;
  speaker: string;
  text: string;
  /** Drafts are provisional and will be replaced by a final. */
  draft: boolean;
  /** Set when the speaker is an unnamed voice, so naming it can update here. */
  slot: number | null;
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  stopped: "остановлен",
  failed: "ошибка",
  noFrames: "нет кадров",
  silent: "тишина",
  ok: "работает",
};

/**
 * Why a stream is unusable, and what to do about it.
 *
 * `silent` is the one that matters: macOS refuses system-audio capture by
 * returning an endless stream of zeroes rather than an error, so this is the
 * only place the user ever learns the permission is missing.
 */
export function diagnose(status: StreamStatus): string | null {
  switch (status.verdict) {
    case "failed":
      return status.error ?? "источник не запустился";
    case "noFrames":
      return "поток запущен, но устройство не прислало ни одного кадра — проверьте, не занято ли оно другим приложением";
    case "silent":
      return status.kind === "system"
        ? "кадры идут, но все сэмплы нулевые. Так macOS отказывает в захвате системного звука — вместо ошибки он отдаёт тишину. Разрешите приложению запись звука в Системных настройках → Конфиденциальность и безопасность."
        : "кадры идут, но все сэмплы нулевые — микрофон заглушён или ему не выдан доступ";
    default:
      return null;
  }
}

export function formatOffset(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}`;
}

/** How to label a turn: a name, a numbered voice, or the stream it came from. */
export function speakerLabel(
  stream: StreamName,
  name: string | null,
  slot: number | null,
): string {
  if (name) return name;
  if (slot !== null) return `Спикер ${slot}`;
  return stream === "mic" ? "Вы" : "Собеседник";
}

export function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds} с`;
  return `${Math.floor(seconds / 60)} мин ${String(seconds % 60).padStart(2, "0")} с`;
}

/** Mirrors `models::CatalogueEntry`. */
export interface CatalogueEntry {
  id: string;
  kind: "recognizer" | "vad" | "speakerEmbedding";
  displayName: string;
  summary: string;
  languages: string;
  downloadBytes: number;
  installed: boolean;
  installing: boolean;
  selected: boolean;
  /** Whether the checksum is published by the release or was recorded here. */
  checksumPublished: boolean;
}

/** Mirrors `models::InstallProgress`. */
export type InstallProgress =
  | { state: "downloading"; id: string; downloaded: number; total: number }
  | { state: "installed"; id: string }
  | { state: "cancelled"; id: string }
  | { state: "failed"; id: string; message: string };

export interface Settings {
  recognizerId: string;
  showDrafts: boolean;
  suppressEcho: boolean;
  /** Days to keep meeting audio; zero keeps none. */
  audioRetentionDays: number;
}

/** Mirrors `StorageUsage`. */
export interface StorageUsage {
  audioBytes: number;
  retentionDays: number;
  meetings: number;
}

export const KIND_LABEL: Record<CatalogueEntry["kind"], string> = {
  recognizer: "Распознавание речи",
  vad: "Границы реплик",
  speakerEmbedding: "Различение голосов",
};

export function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} ГБ` : `${Math.round(mb)} МБ`;
}

/** Mirrors `MeetingSummary`. */
export interface MeetingSummary {
  id: number;
  title: string;
  startedAt: string;
  endedAt: string | null;
  mdPath: string | null;
  segmentCount: number;
}

/** Mirrors `ArchivedLine`. */
export interface ArchivedLine {
  id: number;
  stream: StreamName;
  startMs: number;
  speaker: string | null;
  slot: number | null;
  text: string;
}

/** Mirrors `SearchResult`. */
export interface SearchResult {
  meetingId: number;
  meetingTitle: string;
  meetingStartedAt: string;
  startMs: number;
  speaker: string | null;
  slot: number | null;
  snippet: string;
}

/** The markers `store::MATCH_OPEN` / `MATCH_CLOSE` put around matched words. */
const MATCH_OPEN = "\u0002";
const MATCH_CLOSE = "\u0003";

/**
 * Split a snippet into plain and matched runs.
 *
 * Returned as data rather than HTML on purpose: transcript text is whatever
 * people said, and rendering it as markup would make a spoken tag a rendering
 * decision.
 */
export function splitSnippet(snippet: string): { text: string; match: boolean }[] {
  const parts: { text: string; match: boolean }[] = [];
  let rest = snippet;

  while (rest.length > 0) {
    const open = rest.indexOf(MATCH_OPEN);
    if (open === -1) {
      parts.push({ text: rest, match: false });
      break;
    }
    if (open > 0) parts.push({ text: rest.slice(0, open), match: false });

    const close = rest.indexOf(MATCH_CLOSE, open);
    if (close === -1) {
      parts.push({ text: rest.slice(open + 1), match: true });
      break;
    }
    parts.push({ text: rest.slice(open + 1, close), match: true });
    rest = rest.slice(close + 1);
  }

  return parts.filter((part) => part.text.length > 0);
}

/** "2026-08-27T10:03:00+03:00[Europe/Moscow]" -> "27.08.2026, 10:03" */
export function formatMeetingStart(value: string): string {
  const iso = value.replace(/\[.*\]$/, "");
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ru", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
