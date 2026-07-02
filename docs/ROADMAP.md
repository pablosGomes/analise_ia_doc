# Roadmap — analise_ia_doc

Baseado em `docs/documentacao_deteccao_fraude.docx`.

## Fase 0 — Estruturação (concluída)
- Estrutura de pastas, requirements.txt, config de exemplo, template de termo
  de consentimento LGPD, deploy inicial na VPS.

## Fase 1 — Dados (parcialmente pulada por decisão do usuário em 2026-07-02)
- Consentimento formal e anonimização foram pulados para o lote atual de
  documentos (risco assumido pelo usuário). `datasets/legitimos/` contém
  documentos reais não anonimizados — protegidos por `.gitignore`, nunca
  devem ser commitados/enviados para um remote público.

## Fase 2 — Geração de fraude (em andamento)
- Implementar as 6 técnicas descritas em `scripts/geracao_fraude/README.md`.
- Gerar ao menos 10 variantes rotuladas por documento legítimo.

## Fase 3 — Features (Fase A/B da documentação técnica)
- Fase A: embeddings via DinoV2 pré-treinado (`scripts/features/embeddings_dinov2.py`).
- Fase B: mapa de NoisePrint/TruFor como segundo canal (`scripts/features/noiseprint.py`).

## Fase 4 — Validação de dados
- Checksum de CPF (reaproveitar de pablo-servico-documentos-manipulados), MRZ e QR code.

## Fase 5 — Treino e avaliação
- Cabeça de classificação leve sobre embeddings (baseline).
- Validação cruzada contra técnica de fraude não usada no treino.

## Fase 6 — Integração
- Exportar checkpoint para `pablo-servico-documentos-manipulados`.

## Fase 7 (opcional) — Privacy-preserving
- Migrar para classificação por patches anonimizados (referência: FakeIDet),
  se o volume de dados/voluntários crescer.
