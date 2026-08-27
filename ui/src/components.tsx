import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  type StreamStatus,
  type Turn,
  VERDICT_LABEL,
  diagnose,
  formatOffset,
} from "./types";

/** Meters jump instantly and fall away, so a quiet moment reads as quiet. */
const DECAY = 0.72;

function useDecayingLevel(peak: number, active: boolean): number {
  const [level, setLevel] = useState(0);
  useEffect(() => {
    setLevel((previous) => (active ? Math.max(peak, previous * DECAY) : 0));
  }, [peak, active]);
  return level;
}

export function StreamCard({
  title,
  subtitle,
  status,
  compact,
}: {
  title: string;
  subtitle: string;
  status: StreamStatus;
  compact: boolean;
}) {
  const running = status.verdict !== "stopped" && status.verdict !== "failed";
  const level = useDecayingLevel(status.peak, running);
  const problem = diagnose(status);
  const seconds = status.sampleRate ? status.framesCaptured / status.sampleRate : 0;

  return (
    <section className={`card verdict-${status.verdict} ${compact ? "compact" : ""}`}>
      <header>
        <div>
          <h2>{title}</h2>
          {!compact && <p className="subtitle">{subtitle}</p>}
        </div>
        <span className="badge">{VERDICT_LABEL[status.verdict]}</span>
      </header>

      <div className="meter" role="meter" aria-valuenow={Math.round(level * 100)}>
        <div className="meter-fill" style={{ width: `${Math.min(level, 1) * 100}%` }} />
      </div>

      {!compact && (
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
      )}

      {problem && <p className="problem">{problem}</p>}
    </section>
  );
}

export function Transcript({ turns }: { turns: Turn[] }) {
  const pane = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Follow the conversation, but stop fighting the user the moment they scroll
  // up to re-read something.
  useLayoutEffect(() => {
    const element = pane.current;
    if (element && pinned.current) {
      element.scrollTop = element.scrollHeight;
    }
  }, [turns]);

  const onScroll = () => {
    const element = pane.current;
    if (!element) return;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    pinned.current = distance < 48;
  };

  if (turns.length === 0) {
    return (
      <div className="transcript empty">
        <p>Транскрипт появится здесь по мере распознавания.</p>
      </div>
    );
  }

  return (
    <div className="transcript" ref={pane} onScroll={onScroll}>
      {turns.map((turn, index) => {
        // Consecutive turns by one speaker read as one block, the same way the
        // Markdown renders them.
        const previous = turns[index - 1];
        const continues =
          previous?.speaker === turn.speaker && previous?.draft === turn.draft;

        return (
          <p
            key={turn.key}
            className={`turn stream-${turn.stream} ${turn.draft ? "draft" : ""} ${
              continues ? "continues" : ""
            }`}
          >
            {!continues && (
              <span className="turn-head">
                <span className="time">{formatOffset(turn.startMs)}</span>
                <span className="who">{turn.speaker}</span>
              </span>
            )}
            <span className="said">{turn.text}</span>
          </p>
        );
      })}
    </div>
  );
}
