import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { type LearnedCorrection } from "./types";

/**
 * Every repair the transcript has been taught, and a way to add or drop one.
 *
 * The list is the point, not the form. A rule that fires on a word it should
 * not is invisible until you can see the rules, and the measurement that led
 * here — `docs/transcription.md` — was stopped once already by a repair that
 * looked reasonable in isolation and turned "полка" into "Kafka".
 *
 * `hits` is shown for the same reason: a rule nobody has needed since it was
 * taught is a candidate for deletion, and a rule firing constantly is the
 * vocabulary this user actually speaks.
 */
export function Corrections({ onClose }: { onClose: () => void }) {
  const [rules, setRules] = useState<LearnedCorrection[]>([]);
  const [wrong, setWrong] = useState("");
  const [right, setRight] = useState("");
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    invoke<LearnedCorrection[]>("corrections").then(setRules);
  }, []);

  const teach = async () => {
    if (!wrong.trim() || !right.trim()) return;
    setNote(null);
    try {
      setRules(
        await invoke<LearnedCorrection[]>("teach_correction", { wrong, right }),
      );
      setWrong("");
      setRight("");
    } catch (error) {
      setNote(String(error));
    }
  };

  const forget = async (id: number) => {
    setRules(await invoke<LearnedCorrection[]>("forget_correction", { id }));
  };

  return (
    <section className="review">
      <header>
        <div>
          <h2>Словарь исправлений</h2>
          <p className="subtitle">
            Правьте реплики в архиве — исправления попадают сюда сами и чинят
            это слово во всех расшифровках, включая прошлые.
          </p>
        </div>
        <button className="ghost" onClick={onClose}>
          Закрыть
        </button>
      </header>

      {note && <p className="notice">{note}</p>}

      <div className="toast-row">
        <input
          value={wrong}
          onChange={(event) => setWrong(event.target.value)}
          placeholder="Как слышится"
          aria-label="Как распознаётся"
        />
        <input
          value={right}
          onChange={(event) => setRight(event.target.value)}
          placeholder="Как писать"
          aria-label="Как должно быть написано"
        />
        <button className="ghost" onClick={teach} disabled={!wrong.trim() || !right.trim()}>
          Добавить
        </button>
      </div>

      {rules.length === 0 ? (
        <p className="subtitle">
          Пока пусто. Первое исправление реплики заведёт первое правило.
        </p>
      ) : (
        <ul className="voices">
          {rules.map((rule) => (
            <li key={rule.id}>
              <div className="voice-head">
                <strong>
                  {rule.heard} → {rule.right}
                </strong>
                <span className="voice-meta">
                  {rule.hits === 0 ? "ещё не пригодилось" : `применено ${rule.hits} раз`}
                </span>
              </div>
              <div className="toast-row">
                <button className="ghost" onClick={() => forget(rule.id)}>
                  Забыть
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
