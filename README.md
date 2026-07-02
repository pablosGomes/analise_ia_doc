# analise_ia_doc

Projeto de detecção de fraude em documentos de identidade brasileiros via Machine Learning.

Este projeto substitui a abordagem anterior (ELA/Noise/Clone) por um pipeline baseado em:

- Documentos reais legítimos, obtidos com consentimento explícito dos titulares (LGPD).
- Geração de exemplos de fraude a partir dos próprios documentos legítimos (face swap, edição de campos, crop-and-move, recaptura de tela/impressão), nunca a partir de documentos falsos de terceiros.
- Forense de imagem moderna (NoisePrint/TruFor, dupla compressão JPEG) em vez de ELA.
- Modelos de fundação (DinoV2) e arquiteturas híbridas CNN-Transformer (referência: EdgeDoc) para classificação.
- Validação complementar de dados (checksum de CPF, MRZ, QR code).

## Estrutura

```
analise_ia_doc/
├── docs/                           # Documentação técnica e de pesquisa
│   └── documentacao_deteccao_fraude.docx
├── datasets/                       # Documentos legítimos (sempre anonimizados/pseudonimizados) e variantes de fraude geradas
├── scripts/                        # Pipeline: geração de fraude, extração de features, treino, validação de dados
└── README.md
```

## Documentação

Ver `docs/documentacao_deteccao_fraude.docx` para a fundamentação técnica completa: diferença entre documento legítimo e falsificado, técnicas de identificação, ferramentas/datasets existentes e a estratégia de treino recomendada.

## LGPD

Todo documento real usado neste projeto requer consentimento explícito e por escrito do titular, com finalidade específica de treinamento de modelo de ML. Nenhum documento de terceiro sem consentimento deve ser incluído. Ver Seção 6 da documentação técnica para detalhes.

## VPS

Deploy em `/root/apps/analise_ia_doc` (mesmo padrão do projeto anterior, `prototipo_Ia_doc`).
