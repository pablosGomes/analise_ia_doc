# scripts/geracao_fraude/

Gera exemplos de fraude rotulados a partir de documentos legítimos, nunca a
partir de documentos de terceiros. O foco é nas **duas fraudes que fraudadores de
fato usam** em documentos de identidade — as demais técnicas (recorte-e-move,
recaptura, reimpressão) foram removidas por serem pouco representativas do mundo
real; o objetivo é fazer o simples muito bem feito.

## Técnicas

- `troca_foto.py` — substituição da foto de rosto. Alinhamento por landmarks
  faciais (`scripts/comum/rosto.py`, MediaPipe) + harmonização de cor/nitidez;
  variante evidente (colagem simples) para o par fácil × difícil. Vem dos
  documentos reais (o BID tem a foto de rosto apagada).
- `edicao_campos.py` — edição de nome, data de nascimento, CPF ou filiação via
  inpainting + reescrita, com variantes "bem feita" (fonte plausível) e "mal
  feita" (fonte diferente). Roda no BID (caixas ground-truth) e nos reais (OCR).

A corrupção de dígito de CPF continua existindo como conceito, mas na **camada de
validação de dados** (checksum, `scripts/comum/validacao.py`), não como uma
imagem de treino.

## Orquestradores

- `gerar_dataset_bid.py` — gera a partir do BID (sintético): edição de campos + legítimos.
- `gerar_dataset_reais.py` — gera a partir dos documentos reais: troca de foto + edição + legítimos.

Cada amostra passa pelo simulador de captura compartilhado (`saida.finalizar`) e
grava um manifesto (`.manifesto.json`) com a origem e a manipulação aplicada, em
`datasets/gerado/{bid,reais}_{fraude,legitimos}/`.
