import type { Variante } from "../lib/formato";
import { fmt } from "../lib/formato";

const CLASSE_TAG: Record<Variante, string> = {
  ok: "t-ok",
  info: "t-info",
  warn: "t-warn",
  bad: "t-bad",
};

interface RowProps {
  variante: Variante;
  rotuloTag: string;
  titulo: string;
  descricao?: string;
  valor?: number | null;
  esmaecido?: boolean;
}

/** Uma linha de resultado: etiqueta de estado + título + descrição + nota.
 * É o bloco repetido no veredito e nos testes de generalização. */
export function Row({ variante, rotuloTag, titulo, descricao, valor, esmaecido }: RowProps) {
  return (
    <div className="row">
      <span className={`tag ${CLASSE_TAG[variante]}`}>{rotuloTag}</span>
      <div className="body">
        <div className="title">{titulo}</div>
        {descricao && <div className="desc">{descricao}</div>}
      </div>
      {valor !== undefined && (
        <span className={`val${esmaecido ? " fade" : ""}`}>{fmt(valor)}</span>
      )}
    </div>
  );
}
