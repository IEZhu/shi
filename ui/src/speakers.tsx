import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { type UnnamedVoice, formatDuration } from "./types";

/**
 * Offers to name a voice the moment it is first heard.
 *
 * Deliberately a corner card and not a dialog: the user is in a meeting and
 * needs to listen, so ignoring this must cost nothing. Anything not named here
 * comes back on the review screen afterwards.
 *
 * It also never takes keyboard focus. Naming a voice enrols a voiceprint that
 * will label *future* meetings, so a stray keystroke landing in an input that
 * grabbed focus mid-call would plant a wrong identity that outlives the
 * mistake. The user has to click into the field to name someone.
 */
export function SpeakerToast({
  slot,
  onName,
  onDismiss,
}: {
  slot: number;
  onName: (name: string) => Promise<void>;
  onDismiss: () => void;
}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!name.trim() || busy) return;
    setBusy(true);
    try {
      await onName(name.trim());
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="toast">
      <p className="toast-title">Новый голос — Спикер {slot}</p>
      <div className="toast-row">
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && submit()}
          placeholder="Кто это?"
          aria-label={`Имя для Спикера ${slot}`}
        />
        <button className="start" onClick={submit} disabled={busy || !name.trim()}>
          Назвать
        </button>
        <button className="ghost" onClick={onDismiss} aria-label="Отложить">
          Потом
        </button>
      </div>
    </aside>
  );
}

/**
 * After the meeting: name whatever is still unnamed, in one pass.
 *
 * Each voice is shown with the longest thing it actually said, which
 * identifies a person far faster than a speaker number does.
 */
export function ReviewScreen({
  meetingId,
  onNamed,
  onClose,
}: {
  meetingId: number;
  onNamed: (slot: number, name: string) => void;
  onClose: () => void;
}) {
  const [voices, setVoices] = useState<UnnamedVoice[] | null>(null);
  const [names, setNames] = useState<Record<number, string>>({});
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    invoke<UnnamedVoice[]>("unnamed_voices", { meetingId })
      .then(setVoices)
      .catch((error) => {
        setFailure(String(error));
        setVoices([]);
      });
  }, [meetingId]);

  const save = async (slot: number) => {
    const name = (names[slot] ?? "").trim();
    if (!name) return;
    setBusy(true);
    setFailure(null);
    try {
      await invoke("name_voice", { meetingId, slot, name });
      onNamed(slot, name);
      setVoices((current) => (current ?? []).filter((v) => v.slot !== slot));
    } catch (error) {
      setFailure(String(error));
    } finally {
      setBusy(false);
    }
  };

  if (voices === null) {
    return null;
  }

  return (
    <section className="review">
      <header>
        <div>
          <h2>Кто это говорил?</h2>
          <p className="subtitle">
            {voices.length === 0
              ? "Все голоса опознаны."
              : "Назовите голос — он подпишется во всей встрече и будет узнан в следующий раз."}
          </p>
        </div>
        <button className="ghost" onClick={onClose}>
          {voices.length === 0 ? "Закрыть" : "Пропустить"}
        </button>
      </header>

      {failure && <p className="notice error">{failure}</p>}

      <ul className="voices">
        {voices.map((voice) => (
          <li key={voice.slot}>
            <div className="voice-head">
              <strong>Спикер {voice.slot}</strong>
              <span className="voice-meta">
                {formatDuration(voice.totalSpeechMs)} · {voice.utterances} реплик
              </span>
            </div>
            {voice.excerpt && <p className="excerpt">«{voice.excerpt}»</p>}
            <div className="toast-row">
              <input
                value={names[voice.slot] ?? ""}
                onChange={(event) =>
                  setNames((current) => ({ ...current, [voice.slot]: event.target.value }))
                }
                onKeyDown={(event) => event.key === "Enter" && save(voice.slot)}
                placeholder="Имя"
                aria-label={`Имя для Спикера ${voice.slot}`}
              />
              <button
                className="start"
                onClick={() => save(voice.slot)}
                disabled={busy || !(names[voice.slot] ?? "").trim()}
              >
                Сохранить
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
