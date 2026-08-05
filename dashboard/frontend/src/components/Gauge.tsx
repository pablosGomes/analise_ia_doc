interface GaugeProps {
  valor: number | null | undefined;
}

/** Barra de 0,50 (chute) a 1,00 (acerto perfeito) com um pino marcando a nota
 * atual. As quatro faixas de cor (cinza/azul/âmbar/vermelho) espelham a
 * tabela "Como ler as notas". */
export function Gauge({ valor }: GaugeProps) {
  const v = valor ?? 0.5;
  const posicao = Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100;
  return (
    <div className="gauge">
      <div className="track">
        <i style={{ width: "10%", background: "#39414d" }} />
        <i style={{ width: "30%", background: "#4c8dff" }} />
        <i style={{ width: "40%", background: "#c8922a" }} />
        <i style={{ width: "20%", background: "#d9544d" }} />
        <span className="pin" style={{ left: `calc(${posicao}% - 1px)` }} />
      </div>
      <div className="ticks">
        <span>0,50 chute</span>
        <span>1,00 perfeito</span>
      </div>
    </div>
  );
}
