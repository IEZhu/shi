import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { StreamCard, Transcript } from "./components";
import { useTranscript } from "./useTranscript";
import type { Readiness, StreamStatus } from "./types";

function idle(kind: "mic" | "system"): StreamStatus {
  return {
    kind,
    verdict: "stopped",
    deviceName: null,
    sampleRate: null,
    channels: null,
    framesCaptured: 0,
    framesDropped: 0,
    peak: 0,
    error: null,
  };
}

export function App() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { turns, reset } = useTranscript();
  const unlisten = useRef<(() => void) | null>(null);

  useEffect(() => {
    invoke<Readiness>("readiness").then(setReadiness).catch(() => {});
    listen<Readiness>("readiness", ({ payload }) => setReadiness(payload)).then(
      (stop) => {
        unlisten.current = stop;
      },
    );
    return () => unlisten.current?.();
  }, []);

  const recording = readiness?.recording ?? false;
  const checking =
    !recording &&
    readiness !== null &&
    (readiness.mic.verdict !== "stopped" || readiness.system.verdict !== "stopped");
  const active = recording || checking;
  const modelsMissing = (readiness?.missingModels.length ?? 0) > 0;

  const act = useCallback(
    async (command: string, args?: Record<string, unknown>) => {
      setBusy(true);
      setFailure(null);
      try {
        setReadiness(await invoke<Readiness>(command, args));
      } catch (error) {
        setFailure(String(error));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const startMeeting = useCallback(async () => {
    reset();
    await act("start_meeting", { title });
  }, [act, reset, title]);

  return (
    <main className={active ? "active" : ""}>
      <header className="top">
        <div className="identity">
          <h1>{recording ? "Идёт запись" : "Проверка захвата"}</h1>
          <p className="lede">
            {recording
              ? "Микрофон и системный вывод распознаются независимо."
              : "Убедитесь, что оба потока живые, до начала встречи."}
          </p>
        </div>

        <div className="controls">
          {!active && (
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Название встречи"
              aria-label="Название встречи"
            />
          )}
          {active ? (
            <button className="stop" onClick={() => act("stop")} disabled={busy}>
              Остановить
            </button>
          ) : (
            <>
              <button className="ghost" onClick={() => act("start_check")} disabled={busy}>
                Проверить
              </button>
              <button
                className="start"
                onClick={startMeeting}
                disabled={busy || modelsMissing}
                title={modelsMissing ? "Сначала скачайте модели" : undefined}
              >
                Начать запись
              </button>
            </>
          )}
        </div>
      </header>

      {modelsMissing && (
        <p className="notice">
          Не хватает моделей: {readiness?.missingModels.join(", ")}. Запустите{" "}
          <code>scripts/fetch-models.sh</code>.
        </p>
      )}
      {failure && <p className="notice error">{failure}</p>}

      <div className="streams">
        <StreamCard
          title="Микрофон"
          subtitle="вы — один голос, диаризация не нужна"
          status={readiness?.mic ?? idle("mic")}
          compact={recording}
        />
        <StreamCard
          title="Системный звук"
          subtitle="удалённые участники — здесь работает диаризация"
          status={readiness?.system ?? idle("system")}
          compact={recording}
        />
      </div>

      {recording && (
        <>
          <div className="transcript-head">
            <h2>Транскрипт</h2>
            {readiness?.rtf != null && readiness.rtf > 0 && (
              <span className="rtf" title="во сколько раз быстрее реального времени">
                ×{(1 / readiness.rtf).toFixed(1)}
              </span>
            )}
          </div>
          <Transcript turns={turns} />
        </>
      )}
    </main>
  );
}
