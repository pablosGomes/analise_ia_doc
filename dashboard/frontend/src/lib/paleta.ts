/** Cores das séries nos gráficos, na ordem em que as linhas são adicionadas
 * (nota geral primeiro, depois uma cor por técnica). Espelha os acentos
 * usados no CSS (--accent, --ok, etc.) para os elementos que o CSS puro não
 * alcança (os `stroke` do recharts são inline). */
export const CORES_SERIE = [
  "#4c8dff", // --accent
  "#2ea56b", // --ok
  "#a78bfa", // roxo
  "#c8922a", // --warn
  "#d9544d", // --bad
  "#22b8cf", // ciano
] as const;

export function corDaSerie(indice: number): string {
  return CORES_SERIE[indice % CORES_SERIE.length];
}
