import type { EventoEstadoGerador } from "../types";
import { nome } from "../lib/nomes";
import { Card } from "./Card";

interface BaseTreinoProps {
  gerador: EventoEstadoGerador | null;
}

/** Tabela de contagens com barra proporcional ao maior valor, ordenada do
 * maior para o menor. */
function TabelaContagem({ contagens }: { contagens: Record<string, number> }) {
  const entradas = Object.entries(contagens).sort((a, b) => b[1] - a[1]);
  if (!entradas.length) {
    return (
      <table>
        <tbody>
          <tr>
            <td className="muted">—</td>
          </tr>
        </tbody>
      </table>
    );
  }
  const maior = Math.max(...entradas.map(([, v]) => v));
  return (
    <table>
      <tbody>
        {entradas.map(([chave, valor]) => (
          <tr key={chave}>
            <td>
              {nome(chave)}
              <div className="meter">
                <i style={{ width: `${(100 * valor) / maior}%` }} />
              </div>
            </td>
            <td className="r">{valor}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Volume e composição do dataset usado na última rodada de treino. */
export function BaseTreino({ gerador }: BaseTreinoProps) {
  if (!gerador) {
    return (
      <Card titulo="Base de treino">
        <div className="empty">sem dados</div>
      </Card>
    );
  }

  return (
    <Card
      titulo="Base de treino"
      lead="Volume e composição das amostras. Cada técnica de geração produz a mesma fraude por um caminho diferente — usamos várias de propósito, para o modelo não decorar nenhuma."
    >
      <div className="stats">
        <div className="stat">
          <div className="n">{gerador.n_amostras}</div>
          <div className="l">amostras</div>
        </div>
        <div className="stat">
          <div className="n">{gerador.n_fraude}</div>
          <div className="l">falsificadas</div>
        </div>
        <div className="stat">
          <div className="n">{gerador.n_legitimo}</div>
          <div className="l">legítimas</div>
        </div>
      </div>
      <div className="grid" style={{ gap: 24 }}>
        <div>
          <div className="sub-h">Fraudes por técnica de geração</div>
          <TabelaContagem contagens={gerador.por_gerador} />
        </div>
        <div>
          <div className="sub-h">Origem das amostras</div>
          <TabelaContagem contagens={gerador.por_fonte} />
        </div>
      </div>
    </Card>
  );
}
