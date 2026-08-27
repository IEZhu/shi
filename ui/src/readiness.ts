/// Mirrors `capture::StreamStatus` on the Rust side.
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
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  stopped: "остановлен",
  failed: "ошибка",
  noFrames: "нет кадров",
  silent: "тишина",
  ok: "работает",
};

/**
 * Why a stream is not usable, and what to do about it.
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
