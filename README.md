# analise_ia_doc

Projeto de detecção de fraude em documentos de identidade brasileiros via Machine Learning.

Este projeto substitui a abordagem anterior (ELA/Noise/Clone) por um pipeline baseado em:

- Documentos reais legítimos, obtidos com consentimento explícito dos titulares (LGPD).
- Geração de exemplos de fraude a partir dos próprios documentos legítimos — foco nas duas fraudes mais usadas: **troca de foto de rosto** (alinhada por landmarks) e **edição de campos** — nunca a partir de documentos falsos de terceiros.
- Forense de imagem moderna (NoisePrint/TruFor, dupla compressão JPEG) em vez de ELA.
- Modelos de fundação (DinoV2) e arquiteturas híbridas CNN-Transformer (referência: EdgeDoc) para classificação.
- Validação complementar de dados (checksum de CPF, MRZ, QR code).

## Estrutura

```
analise_ia_doc/
├── docs/
│   ├── documentacao_deteccao_fraude.docx   # Memória oficial do projeto
│   ├── ROADMAP.md · ROTEIRO_ESTUDOS.md
│   └── melhoria_das_falsificacoes.md
├── datasets/                       # (gitignored) FONTE (legitimos, BID) × GERADO (fraude/legítimos)
├── models/                         # (gitignored) modelos baixados (mediapipe, etc.)
├── scripts/
│   ├── comum/                      # utilitários: captura, inpaint, ruido, textura, valores, deteccao, rosto
│   ├── geracao_fraude/             # troca de foto + edição + orquestradores (bid, reais) + manifesto
│   ├── avaliacao/                  # sonda anti-atalho (sanity_atalho.py)
│   ├── features/                   # embeddings DinoV2 (Fase 3)
│   ├── treino/                     # classificador leve (Fase 3)
│   └── validacao_dados/            # checagem de CPF/MRZ/QR (Fase 4)
├── tests/
└── README.md
```

## Documentação

Ver `docs/documentacao_deteccao_fraude.docx` — a **memória oficial** do projeto — para a fundamentação técnica completa: diferença entre documento legítimo e falsificado, técnicas de identificação, ferramentas/datasets existentes, a estratégia de treino recomendada e (Seções 10/10.6/10.7) a geração de fraude v2.

## LGPD

Todo documento real usado neste projeto requer consentimento explícito e por escrito do titular, com finalidade específica de treinamento de modelo de ML. Nenhum documento de terceiro sem consentimento deve ser incluído. Ver Seção 6 da documentação técnica para detalhes.

## VPS

Deploy em `/root/apps/analise_ia_doc` (mesmo padrão do projeto anterior, `prototipo_Ia_doc`).
