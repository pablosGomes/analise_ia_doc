/** Formatos dos eventos publicados por `scripts/treino/treinar_ao_vivo.py`
 * (via `publicador.py`) e retransmitidos por `server.py` no stream SSE.
 * Mantidos em espelho com o payload real emitido pelo backend — qualquer
 * mudança de campo lá precisa ser refletida aqui. */

interface EventoBase {
  run_id: string;
  ts: number;
}

export interface EventoInicio extends EventoBase {
  tipo: "inicio";
  dispositivo: string;
  epocas: number;
  n_treino: number;
  n_val: number;
}

export interface EventoEpoca extends EventoBase {
  tipo: "epoca";
  epoca: number;
  loss_treino: number;
  loss_val: number;
  auc_val: number;
  auc_por_tecnica: Record<string, number>;
}

export interface EventoFim extends EventoBase {
  tipo: "fim";
  epocas: number;
}

/** Snapshot da sonda one-shot: contagens do dataset + as AUCs de diagnóstico
 * (agrupada, leave-one-technique-out, leave-one-generator-out, atalho de
 * origem). Publicado uma vez no início de cada treino. */
export interface EventoEstadoGerador extends EventoBase {
  tipo: "estado_gerador";
  n_amostras: number;
  n_fraude: number;
  n_legitimo: number;
  por_tecnica: Record<string, number>;
  por_gerador: Record<string, number>;
  por_fonte: Record<string, number>;
  auc_fonte_sozinha: number | null;
  equilibrado_por_origem: boolean;
  auc_agrupada: number;
  loto: Record<string, number>;
  logo: Record<string, number>;
  separabilidade_fonte: number | null;
}

export type TreinoEvento = EventoInicio | EventoEpoca | EventoFim | EventoEstadoGerador;

/** Estado acumulado de uma run: o evento de início (se já chegou) + a série
 * de épocas recebidas até agora, na ordem de chegada. */
export interface EstadoRun {
  inicio: EventoInicio | null;
  epocas: EventoEpoca[];
}
