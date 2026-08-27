import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  type StreamName,
  type TranscriptEvent,
  type Turn,
  defaultSpeaker,
} from "./types";

interface FinalTurn extends Turn {
  draft: false;
}

/**
 * Live transcript state.
 *
 * Drafts and finals are kept apart because they behave differently: there is at
 * most one draft per stream and it is always superseded, while finals only ever
 * accumulate. Merging them at render time means a draft can never leave a
 * duplicate behind when its final arrives.
 */
export function useTranscript() {
  const [finals, setFinals] = useState<FinalTurn[]>([]);
  const [drafts, setDrafts] = useState<Partial<Record<StreamName, Turn>>>({});
  const unlisten = useRef<(() => void) | null>(null);

  useEffect(() => {
    listen<TranscriptEvent>("transcript", ({ payload }) => {
      switch (payload.kind) {
        case "draft":
          setDrafts((current) => ({
            ...current,
            [payload.stream]: {
              key: `draft-${payload.stream}`,
              stream: payload.stream,
              startMs: payload.startMs,
              speaker: defaultSpeaker(payload.stream),
              text: payload.text,
              draft: true,
            },
          }));
          break;

        case "final":
          setDrafts((current) => {
            const next = { ...current };
            delete next[payload.stream];
            return next;
          });
          setFinals((current) => [
            ...current,
            {
              key: `final-${payload.id}`,
              stream: payload.stream,
              startMs: payload.startMs,
              speaker: payload.speaker ?? defaultSpeaker(payload.stream),
              text: payload.text,
              draft: false,
            },
          ]);
          break;

        case "draftAbandoned":
          setDrafts((current) => {
            const next = { ...current };
            delete next[payload.stream];
            return next;
          });
          break;
      }
    }).then((stop) => {
      unlisten.current = stop;
    });

    return () => unlisten.current?.();
  }, []);

  const reset = useCallback(() => {
    setFinals([]);
    setDrafts({});
  }, []);

  // Drafts sit at the end: they are the words being spoken right now.
  const turns: Turn[] = [
    ...[...finals].sort((a, b) => a.startMs - b.startMs),
    ...Object.values(drafts).filter((t): t is Turn => t !== undefined),
  ];

  return { turns, reset };
}
