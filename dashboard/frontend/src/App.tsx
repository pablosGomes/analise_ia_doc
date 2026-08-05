import { BaseTreino } from "./components/BaseTreino";
import { ComoLerNotas } from "./components/ComoLerNotas";
import { GraficoAuc, GraficoLoss } from "./components/Graficos";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { TestesGeneralizacao } from "./components/TestesGeneralizacao";
import { useTreinoStream } from "./hooks/useTreinoStream";

export default function App() {
  const { conectado, gerador, runAtual } = useTreinoStream();

  return (
    <>
      <Header conectado={conectado} runAtual={runAtual} />
      <main>
        <Hero gerador={gerador} />
        <ComoLerNotas />
        <TestesGeneralizacao gerador={gerador} />
        <BaseTreino gerador={gerador} />
        <div className="grid">
          <GraficoLoss runAtual={runAtual} />
          <GraficoAuc runAtual={runAtual} />
        </div>
      </main>
    </>
  );
}
