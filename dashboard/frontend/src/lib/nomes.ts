/** Tradução das chaves internas (técnica, gerador, fonte) para rótulos que
 * fazem sentido para quem não conhece o código do gerador de fraude. */
const NOMES: Record<string, string> = {
  edicao_campos: "Edição de campos de texto",
  troca_foto: "Troca da foto de rosto",
  render_classico: "Texto reescrito com fonte do sistema",
  transplante_glifo: "Texto montado com glifos do documento",
  landmark_classico: "Rosto alinhado por landmarks",
  inswapper: "Rosto trocado por rede neural",
  rabisco: "Assinatura sintética",
  bid: "BID · dados sintéticos",
  reais: "Documentos reais",
};

export function nome(chave: string): string {
  return NOMES[chave] ?? chave;
}
