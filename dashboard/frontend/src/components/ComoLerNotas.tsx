import { Card } from "./Card";

/** Seção estática: a régua de leitura da nota (AUC), explicada sem jargão. */
export function ComoLerNotas() {
  return (
    <Card
      titulo="Como ler as notas"
      lead="Mostramos ao modelo uma imagem verdadeira e uma falsificada e perguntamos qual é a falsa. A nota é a proporção de vezes que ele acerta."
    >
      <div className="scale">
        <div>
          <span className="k">0,50</span>
          <span className="v">
            <b>Chute.</b> Não distingue as duas.
          </span>
        </div>
        <div>
          <span className="k">0,55–0,70</span>
          <span className="v">
            <b>Sinal fraco.</b> Percebe algo, mas erra muito.
          </span>
        </div>
        <div>
          <span className="k">0,70–0,90</span>
          <span className="v">
            <b>Sinal forte.</b> Pode ser a fraude — ou um atalho.
          </span>
        </div>
        <div>
          <span className="k">&gt; 0,90</span>
          <span className="v">
            <b>Suspeito.</b> Quase sempre indica atalho, não detecção.
          </span>
        </div>
      </div>
    </Card>
  );
}
