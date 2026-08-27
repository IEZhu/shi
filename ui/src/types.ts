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
      speaker: string | null;
      text: string;
    }
  | { kind: "draftAbandoned"; stream: StreamName };

export type StreamName = "mic" | "system";

export interface Turn {
  key: string;
  stream: StreamName;
  startMs: number;
  speaker: string;
  text: string;
  /** Drafts are provisional and will be replaced by a final. */
  draft: boolean;
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

/** Fallback names before diarization has anything to say. */
export function defaultSpeaker(stream: StreamName): string {
  return stream === "mic" ? "Вы" : "Собеседник";
}
