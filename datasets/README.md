# datasets/

Estrutura de dados do projeto. **Todo o conteúdo de imagem é gitignored** (PII /
dados derivados); só este README e os `.gitkeep` são versionados.

## Organização: FONTE × GERADO

```
datasets/
├── legitimos/            # FONTE — documentos reais do Pablo (16 na VPS; cópia local parcial).
│   └── {cnh,rg,passaporte}/   NUNCA apagar. Base real (LGPD: só com consentimento).
├── BID Dataset/          # FONTE — dataset público BID (28.800 docs BR, PII SINTÉTICA).
│                              <id>_in.jpg + <id>_gt_ocr.txt + <id>_gt_segmentation.jpg.
├── gerado/               # GERADO — imagens de treino (APAGÁVEL / regenerável a partir do código).
│   ├── bid_fraude/edicao_campos/   # edição de campos gerada a partir do BID
│   ├── bid_legitimos/<classe>/     # classe negativa (BID + captura compartilhada)
│   ├── reais_fraude/<tecnica>/     # troca de foto + edição, a partir dos reais
│   └── reais_legitimos/<tipo>/     # classe negativa dos reais
└── processed/            # features/embeddings (Fase 3 — DinoV2).
```

## Convenção: cada leva NOVA substitui a anterior

Não acumular datasets de imagens antigos. `gerar_dataset_bid.py` apaga a leva
anterior automaticamente antes de gerar (use `--manter-antigas` para não apagar).
Os manifestos (`*.manifesto.json`) permitem regenerar/auditar qualquer amostra.

## Como (re)gerar

```powershell
# BID (edição de campos) — apaga a leva anterior e gera a nova
python -m scripts.geracao_fraude.gerar_dataset_bid --por-classe 150 --variantes-legit 7
# reais (troca de foto + edição) — ficam só na máquina, nunca versionados
python -m scripts.geracao_fraude.gerar_dataset_reais --variantes-por-documento 10
# sonda anti-atalho
python -m scripts.avaliacao.sanity_atalho --fraude datasets/gerado/bid_fraude --legitimos datasets/gerado/bid_legitimos
```
