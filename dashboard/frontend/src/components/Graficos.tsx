import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EstadoRun } from "../types";
import { corDaSerie } from "../lib/paleta";
import { nome } from "../lib/nomes";
import { Card } from "./Card";

interface GraficosProps {
  runAtual: EstadoRun | null;
}

const ESTILO_EIXO = { fontSize: 10, fill: "#6b7684", fontFamily: "ui-monospace, monospace" };
const ESTILO_TOOLTIP = {
  background: "#161a21",
  border: "1px solid #1e232c",
  borderRadius: 8,
  fontSize: 12,
};

interface Serie {
  chave: string;
  nome: string;
  cor: string;
}

function Legenda({ series }: { series: Serie[] }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.chave}>
          <i style={{ background: s.cor }} />
          {s.nome}
        </span>
      ))}
    </div>
  );
}

/** O `formatter` do Tooltip do recharts recebe a `dataKey` bruta (ex.:
 * "edicao_campos") como nome da série — troca pelo rótulo amigável definido
 * em `series`. */
function formatadorTooltip(series: Serie[]) {
  const porChave = new Map(series.map((s) => [s.chave, s.nome]));
  return (valor: number, chave: string): [string, string] => [valor.toFixed(3), porChave.get(chave) ?? chave];
}

/** Erro de treino (sempre cai) vs. erro de validação — quando a segunda
 * curva sobe, o modelo passou a memorizar em vez de aprender. */
export function GraficoLoss({ runAtual }: GraficosProps) {
  const epocas = runAtual?.epocas ?? [];
  const dados = epocas.map((e) => ({ epoca: e.epoca, treino: e.loss_treino, validacao: e.loss_val }));
  const series: Serie[] = [
    { chave: "treino", nome: "treino", cor: corDaSerie(0) },
    { chave: "validacao", nome: "validação", cor: corDaSerie(1) },
  ];

  return (
    <Card titulo="Aprendizado × memorização" lead="O erro de treino sempre cai. Quando o erro de validação passa a subir, o modelo começou a memorizar.">
      <div className="chart-wrap">
        {dados.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={dados} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1e232c" vertical={false} />
              <XAxis dataKey="epoca" tick={ESTILO_EIXO} tickLine={false} axisLine={{ stroke: "#1e232c" }} label={{ value: "rodada", position: "insideBottomRight", offset: -4, style: ESTILO_EIXO }} />
              <YAxis tick={ESTILO_EIXO} tickLine={false} axisLine={false} width={38} tickFormatter={(v: number) => v.toFixed(2)} />
              <Tooltip contentStyle={ESTILO_TOOLTIP} labelStyle={{ color: "#e8ecf1" }} labelFormatter={(v) => `rodada ${v}`} formatter={formatadorTooltip(series)} />
              {series.map((s) => (
                <Line key={s.chave} type="monotone" dataKey={s.chave} stroke={s.cor} strokeWidth={1.8} dot={false} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty">aguardando o treino</div>
        )}
      </div>
      <Legenda series={series} />
    </Card>
  );
}

/** Nota (AUC) em dados de validação, por rodada — geral e por técnica de
 * fraude. A linha tracejada em 0,50 marca o nível do chute. */
export function GraficoAuc({ runAtual }: GraficosProps) {
  const epocas = runAtual?.epocas ?? [];
  const tecnicas = new Set<string>();
  epocas.forEach((e) => Object.keys(e.auc_por_tecnica ?? {}).forEach((t) => tecnicas.add(t)));

  const dados = epocas.map((e) => ({ epoca: e.epoca, notaGeral: e.auc_val, ...e.auc_por_tecnica }));
  const series: Serie[] = [
    { chave: "notaGeral", nome: "nota geral", cor: corDaSerie(0) },
    ...[...tecnicas].map((t, i) => ({ chave: t, nome: nome(t), cor: corDaSerie(i + 1) })),
  ];

  return (
    <Card titulo="Nota em dados não vistos" lead="Desempenho em imagens que o modelo nunca viu, a cada rodada. A linha tracejada marca o nível do chute.">
      <div className="chart-wrap">
        {dados.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={dados} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1e232c" vertical={false} />
              <XAxis dataKey="epoca" tick={ESTILO_EIXO} tickLine={false} axisLine={{ stroke: "#1e232c" }} label={{ value: "rodada", position: "insideBottomRight", offset: -4, style: ESTILO_EIXO }} />
              <YAxis domain={[0, 1]} tick={ESTILO_EIXO} tickLine={false} axisLine={false} width={38} tickFormatter={(v: number) => v.toFixed(2)} />
              <Tooltip contentStyle={ESTILO_TOOLTIP} labelStyle={{ color: "#e8ecf1" }} labelFormatter={(v) => `rodada ${v}`} formatter={formatadorTooltip(series)} />
              <ReferenceLine y={0.5} stroke="#6b7684" strokeDasharray="3 4" />
              {series.map((s) => (
                <Line key={s.chave} type="monotone" dataKey={s.chave} stroke={s.cor} strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty">aguardando o treino</div>
        )}
      </div>
      <Legenda series={series} />
    </Card>
  );
}
