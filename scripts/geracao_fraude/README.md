# scripts/geracao_fraude/

Gera exemplos de fraude rotulados a partir de cada documento legítimo
(pseudonimizado), nunca a partir de documentos de terceiros. Técnicas previstas
(Seção 5.2 da documentação técnica):

- `troca_foto.py` — substituição da foto de rosto (splicing), com e sem ajuste
  de iluminação/borda.
- `edicao_campos.py` — edição de nome, data de nascimento, CPF ou número de
  registro via inpainting + reescrita, incluindo variantes "bem feitas" e
  "mal feitas".
- `crop_and_move.py` — recorte e reposicionamento de uma região (técnica
  validada pelo dataset acadêmico SIDTD sobre o MIDV-2020).
- `recaptura_tela.py` — exibição em monitor + refotografia, para artefatos
  reais de moiré/reflexo.
- `reimpressao.py` — impressão em papel comum + refotografia (metodologia
  FantasyID).
- `digito_verificador.py` — corrupção isolada do CPF/RG numérico, para treinar
  a camada de validação de dados independentemente da camada de imagem.

Cada script deve gravar a variante gerada em
`datasets/fraude_gerada/<tecnica>/` com um manifesto (`.json`) apontando para o
documento legítimo de origem e o tipo exato de manipulação aplicada.
