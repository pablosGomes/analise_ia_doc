import type { EventoEstadoGerador } from "../types";
import { faixa, fmt } from "../lib/formato";
import { nome } from "../lib/nomes";
import { Gauge } from "./Gauge";
import { Note } from "./Note";
import { Row } from "./Row";

interface HeroProps {
  gerador: EventoEstadoGerador | null;
}

function tituloVeredito(auc: number, separabilidade: number | null): string {
  const bom = auc >= 0.7 && (separabilidade == null || separabilidade < 0.9);
  const meio = auc >= 0.6;
  if (bom) return "O modelo já identifica boa parte das fraudes";
  if (meio) return "O modelo percebe as fraudes, mas com margem de erro alta";
  return "O modelo ainda não distingue documento verdadeiro de falsificado";
}

function descricaoTecnica(tecnica: string, v: number): string {
  if (v < 0.55) {
    return tecnica === "edicao_campos"
      ? "Sem assinatura visual detectável. Esperado: substituir um nome não altera a aparência do documento — a checagem cabe à validação de dados (dígito verificador), não à imagem."
      : "Sem sinal detectável para esta fraude.";
  }
  if (v < 0.7) return "Sinal presente, porém com erro alto. Demanda mais amostras e maior diversidade de geração.";
  if (v < 0.9) return "Detecção consistente para esta fraude.";
  return "Separação quase perfeita — investigar possível atalho.";
}

/** Cartão de abertura: a resposta direta a "o modelo já identifica fraude?",
 * seguida da nota geral, o indicador visual e o resultado por tipo de
 * fraude — tudo calculado a partir da última foto do gerador. */
export function Hero({ gerador }: HeroProps) {
  if (!gerador) {
    return (
      <section className="card hero">
        <div className="empty">Aguardando o primeiro treino.</div>
      </section>
    );
  }

  const { auc_agrupada: auc, separabilidade_fonte: sep, loto, auc_fonte_sozinha: atalho, equilibrado_por_origem: equilibrado } = gerador;

  return (
    <section className="card hero">
      <h1>{tituloVeredito(auc, sep)}</h1>
      <div className="sub">
        Nota geral <b>{fmt(auc)}</b> · {faixa(auc).rotulo}
      </div>
      <Gauge valor={auc} />
      <div className="rows">
        {Object.entries(loto).map(([tecnica, v]) => (
          <Row
            key={tecnica}
            variante={faixa(v).variante}
            rotuloTag={faixa(v).rotulo}
            titulo={nome(tecnica)}
            descricao={descricaoTecnica(tecnica, v)}
            valor={v}
          />
        ))}
      </div>

      {equilibrado && atalho != null && (
        <Note variante="ok" titulo="Atalho de origem neutralizado">
          As amostras sintéticas e as de documentos reais são visualmente distintas e tinham proporções
          diferentes de fraude — apenas a origem já rendia nota <b className="num">{fmt(atalho)}</b>. As notas
          acima são medidas com as origens equilibradas, refletindo a fraude e não a procedência.
        </Note>
      )}
      {!equilibrado && sep != null && sep >= 0.9 && (
        <Note variante="bad" titulo="Métricas contaminadas pela origem">
          As origens são separáveis ({fmt(sep)}) e não foram equilibradas: parte do desempenho vem de
          identificar a procedência da imagem.
        </Note>
      )}
    </section>
  );
}
