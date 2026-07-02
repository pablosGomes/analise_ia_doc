# scripts/geracao_fraude/

Gera exemplos de fraude rotulados a partir de cada documento legítimo, nunca a
partir de documentos de terceiros. Técnicas previstas (Seção 5.2 da
documentação técnica):

- `troca_foto.py` — substituição da foto de rosto (splicing).
- `edicao_campos.py` — edição de nome, data de nascimento, CPF ou número de
  registro via inpainting + reescrita.
- `crop_and_move.py` — recorte e reposicionamento de uma região (técnica SIDTD).
- `recaptura_tela.py` — exibição em monitor + refotografia.
- `reimpressao.py` — impressão em papel comum + refotografia (FantasyID).
- `digito_verificador.py` — corrupção isolada do CPF/RG numérico.

Cada script deve gravar a variante gerada em
`datasets/fraude_gerada/<tecnica>/` com um manifesto (`.json`) apontando para o
documento legítimo de origem.
