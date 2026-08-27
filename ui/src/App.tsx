import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  type Readiness,
  type StreamStatus,
  VERDICT_LABEL,
  diagnose,
} from "./readiness";

/** Meters jump instantly and fall away, so a quiet moment reads as quiet. */
const DECAY = 0.72;

function useDecayingLevel(peak: number): number {
  const [level, setLevel] = useState(0);
  useEffect(() => {
    setLevel((previous) => Math.max(peak, previous * DECAY));
  }, [peak]);
  return level;
}

function Meter({ peak, active }: { peak: number; active: boolean }) {
  const level = useDecayingLevel(active ? peak : 0);
  return (
    <div className="meter" role="meter" aria-valuenow={Math.round(level * 100)}>
      <div className="meter-fill" style={{ width: `${Math.min(level, 1) * 100}%` }} />
    </div>
  );
}

function StreamCard({ title, subtitle, status }: {
  title: string;
  subtitle: string;
  status: StreamStatus;
}) {
  const problem = diagnose(status);
  const running = status.verdict !== "stopped" && status.verdict !== "failed";
  const seconds = status.sampleRate
    ? status.framesCaptured / status.sampleRate
    : 0;

  return (
    <section className={`card verdict-${status.verdict}`}>
      <header>
        <div>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
        </div>
        <span className="badge">{VERDICT_LABEL[status.verdict]}</span>
      </header>

      <Meter peak={status.peak} active={running} />

      <dl className="facts">
        <div>
          <dt>Устройство</dt>
          <dd>{status.deviceName ?? "—"}</dd>
        </div>
        <div>
          <dt>Формат</dt>
          <dd>
            {status.sampleRate
              ? `${(status.sampleRate / 1000).toFixed(1)} кГц · ${status.channels} кан.`
              : "—"}
          </dd>
        </div>
        <div>
          <dt>Захвачено</dt>
          <dd>{running ? `${seconds.toFixed(1)} с` : "—"}</dd>
        </div>
        <div>
          <dt>Потеряно</dt>
          <dd className={status.framesDropped > 0 ? "warn" : undefined}>
            {running ? status.framesDropped.toLocaleString("ru") : "—"}
          </dd>
        </div>
      </dl>

      {problem && <p className="problem">{problem}</p>}
    </section>
  );
}

export function App() {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [busy, setBusy] = useState(false);
  const unlisten = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Pick up the current state on mount, so a reloaded window is not blank
    // while capture is already running.
    invoke<Readiness>("readiness").then(setReadiness).catch(() => {});

    listen<Readiness>("readiness", (event) => setReadiness(event.payload)).then(
      (stop) => {
        unlisten.current = stop;
      },
    );
    return () => unlisten.current?.();
  }, []);

  const running =
    readiness !== null &&
    (readiness.mic.verdict !== "stopped" || readiness.system.verdict !== "stopped");

  const toggle = useCallback(async () => {
    setBusy(true);
    try {
      const command = running ? "stop_capture" : "start_capture";
      setReadiness(await invoke<Readiness>(command));
    } finally {
      setBusy(false);
    }
  }, [running]);

  return (
    <main>
      <header className="top">
        <div>
          <h1>Проверка захвата</h1>
          <p className="lede">
            Микрофон и системный вывод пишутся как два независимых потока. Убедитесь,
            что оба живые, <em>до</em> начала встречи.
          </p>
        </div>
        <button onClick={toggle} disabled={busy} className={running ? "stop" : "start"}>
          {running ? "Остановить" : "Проверить"}
        </button>
      </header>

      <div className="grid">
        <StreamCard
          title="Микрофон"
          subtitle="вы — один голос, диаризация не нужна"
          status={readiness?.mic ?? idle("mic")}
        />
        <StreamCard
          title="Системный звук"
          subtitle="удалённые участники — здесь работает диаризация"
          status={readiness?.system ?? idle("system")}
        />
      </div>
    </main>
  );
}

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
