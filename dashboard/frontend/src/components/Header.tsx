import type { EstadoRun } from "../types";

interface HeaderProps {
  conectado: boolean;
  runAtual: EstadoRun | null;
}

export function Header({ conectado, runAtual }: HeaderProps) {
  const texto = runAtual
    ? `${runAtual.epocas.length}${runAtual.inicio ? ` / ${runAtual.inicio.epocas}` : ""} rodadas`
    : "sem treino ativo";
  return (
    <header>
      <div className="hd">
        <div className="brand">
          iadoc <span>· detecção de fraude documental</span>
        </div>
        <div className="status">
          <span className={`dot${conectado ? " on" : ""}`} />
          <span>{texto}</span>
        </div>
      </div>
    </header>
  );
}
