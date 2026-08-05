import type { EventoEstadoGerador } from "../types";
import { AMOSTRA_MINIMA, faixa } from "../lib/formato";
import { nome } from "../lib/nomes";
import { Card } from "./Card";
import { Note } from "./Note";
import { Row } from "./Row";

interface TestesGeneralizacaoProps {
  gerador: EventoEstadoGerador | null;
}

/** Uma linha de teste surpresa: se a contagem de amostras da chave estiver
 * abaixo do mínimo, a nota é marcada como não conclusiva em vez de
 * classificada normalmente — amostra pequena não sustenta veredito. */
function ItemTeste({ chave, valor, contexto, contagem }: { chave: string; valor: number; contexto: string; contagem?: number }) {
  const poucaAmostra = contagem != null && contagem < AMOSTRA_MINIMA;
  const f = poucaAmostra ? { variante: "info" as const, rotulo: "amostra pequena" } : faixa(valor);
  const descricao = poucaAmostra ? `${contexto} · apenas ${contagem} amostras, resultado não conclusivo` : contexto;
  return (
    <Row variante={f.variante} rotuloTag={f.rotulo} titulo={nome(chave)} descricao={descricao} valor={valor} esmaecido={poucaAmostra} />
  );
}

/** Removendo um tipo de fraude ou uma técnica de geração do treino e
 * testando justamente nele: mede se o modelo generalizou ou decorou. */
export function TestesGeneralizacao({ gerador }: TestesGeneralizacaoProps) {
  if (!gerador) {
    return (
      <Card titulo="Testes de generalização">
        <div className="empty">sem dados</div>
      </Card>
    );
  }

  const { logo, loto, por_gerador: porGerador, por_tecnica: porTecnica } = gerador;
  const geradoresComAmostra = Object.entries(logo).filter(([k]) => (porGerador[k] ?? 0) >= AMOSTRA_MINIMA);
  let conclusao: { ok: boolean } | null = null;
  if (geradoresComAmostra.length >= 2) {
    const valores = geradoresComAmostra.map(([, v]) => v);
    const amplitude = Math.max(...valores) - Math.min(...valores);
    conclusao = { ok: amplitude < 0.1 };
  }

  return (
    <Card
      titulo="Testes de generalização"
      lead="Removemos um tipo de fraude — ou uma técnica de geração — do treino e testamos justamente nele. Se o desempenho se mantém, o modelo aprendeu o conceito, não a ferramenta."
    >
      <div className="sub-h">Removendo um tipo de fraude</div>
      <div className="rows">
        {Object.entries(loto).length ? (
          Object.entries(loto).map(([k, v]) => (
            <ItemTeste key={k} chave={k} valor={v} contexto="treinado sem este tipo de fraude" contagem={porTecnica[k]} />
          ))
        ) : (
          <div className="empty">—</div>
        )}
      </div>

      <div className="sub-h">Removendo uma técnica de geração</div>
      <div className="rows">
        {Object.entries(logo).length ? (
          Object.entries(logo).map(([k, v]) => (
            <ItemTeste key={k} chave={k} valor={v} contexto="treinado sem esta técnica" contagem={porGerador[k]} />
          ))
        ) : (
          <div className="empty">—</div>
        )}
      </div>

      {conclusao && (
        <Note variante={conclusao.ok ? "ok" : "warn"} titulo={conclusao.ok ? "Diversidade de geração saudável" : "Uma técnica de geração se destaca"}>
          {conclusao.ok
            ? "As notas das técnicas de geração ficaram próximas — nenhuma delas deixa uma assinatura própria que o modelo possa memorizar."
            : "Uma das técnicas é sensivelmente mais fácil de identificar que as demais, indicando assinatura característica."}{" "}
          Considerando apenas técnicas com ao menos {AMOSTRA_MINIMA} amostras.
        </Note>
      )}
    </Card>
  );
}
