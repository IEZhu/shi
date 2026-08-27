import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  type CatalogueEntry,
  type InstallProgress,
  KIND_LABEL,
  formatBytes,
} from "./types";

/**
 * Downloading, choosing and removing models.
 *
 * The recogniser is a real choice — speed against language coverage — so the
 * entries lead with what actually differs rather than with file sizes.
 */
export function ModelManager({ onClose }: { onClose: () => void }) {
  const [entries, setEntries] = useState<CatalogueEntry[]>([]);
  const [progress, setProgress] = useState<Record<string, InstallProgress>>({});
  const unlisten = useRef<(() => void) | null>(null);

  const refresh = useCallback(async () => {
    setEntries(await invoke<CatalogueEntry[]>("model_catalogue"));
  }, []);

  useEffect(() => {
    refresh();
    listen<InstallProgress>("model", ({ payload }) => {
      setProgress((current) => ({ ...current, [payload.id]: payload }));
      if (payload.state !== "downloading") {
        refresh();
      }
    }).then((stop) => {
      unlisten.current = stop;
    });
    return () => unlisten.current?.();
  }, [refresh]);

  const start = async (id: string) => {
    await invoke("install_model", { id });
    refresh();
  };

  const cancel = async (id: string) => {
    await invoke("cancel_model_install", { id });
  };

  const remove = async (id: string) => {
    await invoke("uninstall_model", { id });
    setProgress((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
    refresh();
  };

  const choose = async (id: string) => {
    const current = await invoke<{ recognizerId: string }>("settings");
    await invoke("save_settings", { settings: { ...current, recognizerId: id } });
    refresh();
  };

  const groups: CatalogueEntry["kind"][] = ["recognizer", "vad", "speakerEmbedding"];

  return (
    <section className="review models">
      <header>
        <div>
          <h2>Модели</h2>
          <p className="subtitle">
            Всё распознавание идёт на этой машине, поэтому модели нужно скачать один раз.
          </p>
        </div>
        <button className="ghost" onClick={onClose}>
          Закрыть
        </button>
      </header>

      {groups.map((kind) => {
        const inGroup = entries.filter((entry) => entry.kind === kind);
        if (inGroup.length === 0) return null;

        return (
          <div key={kind} className="model-group">
            <h3>{KIND_LABEL[kind]}</h3>
            <ul className="voices">
              {inGroup.map((entry) => {
                const state = progress[entry.id];
                const downloading =
                  entry.installing || state?.state === "downloading";
                const fraction =
                  state?.state === "downloading" && state.total > 0
                    ? state.downloaded / state.total
                    : 0;

                return (
                  <li key={entry.id}>
                    <div className="voice-head">
                      <strong>{entry.displayName}</strong>
                      <span className="voice-meta">
                        {formatBytes(entry.downloadBytes)}
                        {entry.installed && " · установлена"}
                        {entry.selected && " · выбрана"}
                      </span>
                    </div>
                    <p className="excerpt">
                      {entry.summary} · {entry.languages}
                    </p>

                    {downloading && (
                      <>
                        <div className="meter">
                          <div
                            className="meter-fill"
                            style={{ width: `${fraction * 100}%` }}
                          />
                        </div>
                        <p className="voice-meta">
                          {state?.state === "downloading"
                            ? `${formatBytes(state.downloaded)} из ${formatBytes(state.total)}`
                            : "начинаем…"}
                        </p>
                      </>
                    )}

                    {state?.state === "failed" && (
                      <p className="notice error">{state.message}</p>
                    )}

                    <div className="toast-row">
                      {downloading ? (
                        <button className="ghost" onClick={() => cancel(entry.id)}>
                          Отменить
                        </button>
                      ) : entry.installed ? (
                        <>
                          {entry.kind === "recognizer" && !entry.selected && (
                            <button className="start" onClick={() => choose(entry.id)}>
                              Выбрать
                            </button>
                          )}
                          <button
                            className="ghost"
                            onClick={() => remove(entry.id)}
                            disabled={entry.selected}
                            title={
                              entry.selected
                                ? "Сначала выберите другую модель"
                                : undefined
                            }
                          >
                            Удалить
                          </button>
                        </>
                      ) : (
                        <button className="start" onClick={() => start(entry.id)}>
                          Скачать
                        </button>
                      )}
                      {!entry.checksumPublished && (
                        <span
                          className="voice-meta"
                          title="Контрольная сумма записана при добавлении модели: она поймает повреждение файла, но не подтверждает подпись издателя."
                        >
                          сумма записана локально
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </section>
  );
}
