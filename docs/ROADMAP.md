# Roadmap — analise_ia_doc

Baseado em `docs/documentacao_deteccao_fraude.docx`.

## Fase 0 — Estruturação (concluída)
- Estrutura de pastas, requirements.txt, config de exemplo, template de termo
  de consentimento LGPD, deploy inicial na VPS.

## Fase 1 — Dados
- Formalizar termo de consentimento assinado para os documentos já obtidos.
- Adaptar o script de anonimização do projeto anterior para
  `scripts/anonimizacao/`.
- Popular `datasets/legitimos/` apenas com documentos pseudonimizados e
  verificados manualmente.

## Fase 2 — Geração de fraude
- Implementar as 6 técnicas descritas em `scripts/geracao_fraude/README.md`.
- Gerar ao menos 10 variantes rotuladas por documento legítimo.

## Fase 3 — Features (Fase A/B da documentação técnica)
- Fase A: embeddings via DinoV2 pré-treinado (`scripts/features/embeddings_dinov2.py`).
- Fase B: mapa de NoisePrint/TruFor como segundo canal (`scripts/features/noiseprint.py`).

## Fase 4 — Validação de dados (Fase C)
- Checksum de CPF, MRZ e QR code, independente do sinal de imagem.

## Fase 5 — Treino e avaliação
- Cabeça de classificação leve sobre embeddings (baseline).
- Validação cruzada contra técnica de fraude não usada no treino
  (mitigação do "Synthetic Utility Gap").

## Fase 6 (opcional) — Privacy-preserving
- Migrar para classificação por patches anonimizados (referência: FakeIDet),
  se o volume de dados/voluntários crescer.
