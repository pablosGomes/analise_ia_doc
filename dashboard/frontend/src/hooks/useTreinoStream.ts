import { useEffect, useRef, useState } from "react";
import type { EstadoRun, EventoEstadoGerador, TreinoEvento } from "../types";

interface EstadoStream {
  /** Conectado ao stream SSE agora (liga o indicador no cabeçalho). */
  conectado: boolean;
  /** Última foto do estado do gerador (contagens do dataset + AUCs de
   * diagnóstico). Publicada uma vez no início de cada treino. */
  gerador: EventoEstadoGerador | null;
  /** A run mais recente a receber um evento, com sua série de épocas. */
  runAtual: EstadoRun | null;
}

/** Conecta ao `/api/stream` (SSE) e mantém o estado do treino ao vivo.
 *
 * O servidor manda primeiro um evento `snapshot` (todo o histórico
 * acumulado, para quem abre o painel no meio de um treino) e depois um
 * evento por vez conforme o treino publica. Os dois casos passam pelo mesmo
 * reducer, então a ordem de chegada nunca diverge do estado final. */
export function useTreinoStream(): EstadoStream {
  const [conectado, setConectado] = useState(false);
  const [gerador, setGerador] = useState<EventoEstadoGerador | null>(null);
  const [runs, setRuns] = useState<Record<string, EstadoRun>>({});
  const [runAtualId, setRunAtualId] = useState<string | null>(null);

  // Evita reabrir a conexão a cada render — só na montagem.
  const processar = useRef((_ev: TreinoEvento) => {});
  processar.current = (ev: TreinoEvento) => {
    if (ev.tipo === "estado_gerador") {
      setGerador(ev);
      return;
    }
    if (ev.tipo === "inicio") {
      setRuns((prev) => ({ ...prev, [ev.run_id]: { inicio: ev, epocas: [] } }));
      setRunAtualId(ev.run_id);
      return;
    }
    if (ev.tipo === "epoca") {
      setRuns((prev) => {
        const atual = prev[ev.run_id] ?? { inicio: null, epocas: [] };
        return { ...prev, [ev.run_id]: { ...atual, epocas: [...atual.epocas, ev] } };
      });
      setRunAtualId(ev.run_id);
    }
    // "fim" não muda nada visualmente além do que as épocas já mostram.
  };

  useEffect(() => {
    const es = new EventSource("/api/stream");

    es.addEventListener("snapshot", (e: MessageEvent<string>) => {
      const eventos = JSON.parse(e.data) as TreinoEvento[];
      eventos.forEach((ev) => processar.current(ev));
    });
    es.onmessage = (e: MessageEvent<string>) => {
      try {
        processar.current(JSON.parse(e.data) as TreinoEvento);
      } catch {
        // heartbeat (": ping") ou evento malformado — ignora
      }
    };
    es.onopen = () => setConectado(true);
    es.onerror = () => setConectado(false);

    return () => es.close();
  }, []);

  const runAtual = runAtualId ? (runs[runAtualId] ?? null) : null;
  return { conectado, gerador, runAtual };
}
