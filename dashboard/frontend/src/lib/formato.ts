/** Nota (AUC) formatada com vírgula decimal, ou travessão quando ausente. */
export function fmt(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(3).replace(".", ",");
}

export type Variante = "ok" | "info" | "warn" | "bad";

export interface Faixa {
  variante: Variante;
  rotulo: string;
}

/** Classifica uma nota (AUC) na escala de leitura do painel.
 * Ver seção "Como ler as notas" — as fronteiras (0,55 / 0,70 / 0,90) são as
 * mesmas usadas na tabela de referência exibida ao usuário. */
export function faixa(v: number | null | undefined): Faixa {
  if (v == null || Number.isNaN(v)) return { variante: "info", rotulo: "—" };
  if (v < 0.55) return { variante: "ok", rotulo: "não distingue" };
  if (v < 0.7) return { variante: "info", rotulo: "sinal fraco" };
  if (v < 0.9) return { variante: "warn", rotulo: "sinal forte" };
  return { variante: "bad", rotulo: "suspeito" };
}

/** Amostras abaixo disso não sustentam conclusão — a nota pode ser ruído
 * estatístico. Usado para marcar "amostra pequena" nos testes de
 * generalização por técnica de geração. */
export const AMOSTRA_MINIMA = 20;
