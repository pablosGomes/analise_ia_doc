# Roadmap — analise_ia_doc

Baseado em `docs/documentacao_deteccao_fraude.docx` (a memória do projeto).
Status atualizado em 2026-07-20.

## Fase 0 — Estruturação ✅ concluída
- Estrutura de pastas, requirements.txt, config de exemplo, template de termo
  de consentimento LGPD, deploy inicial na VPS.

## Fase 1 — Dados ✅ (parcial)
- `datasets/legitimos/` populado com 16 documentos (cnh: 6, rg: 10, passaporte: 0).
- Pendente: formalizar termo de consentimento assinado; adaptar script de
  anonimização para `scripts/anonimizacao/`.

## Fase 2 — Geração de fraude 🔄 em foco
- **Paridade de pipeline** (captura compartilhada) + **sonda anti-atalho** medida
  sobre embeddings DinoV2 com split por documento — a sonda linear antiga dava
  falsa segurança.
- **Refoco (2026-07-20): só as duas fraudes que fraudadores de fato usam** —
  troca de foto de rosto e edição de campos. Removidas recorte-e-move, dígito
  (como imagem), recaptura e reimpressão. Filosofia: fazer o simples muito bem feito.
- **BID integrado** para volume de edição de campos (caixas ground-truth).
- **Troca de foto sendo refeita** com alinhamento por landmarks (`scripts/comum/rosto.py`,
  MediaPipe) — a versão antiga (colagem por caixa) não alinhava o rosto.
- Diversificação de geradores em andamento (clássico + neural), para o modelo
  aprender a fraude e não a assinatura de uma ferramenta só.
- Pendência: validação externa contra fraude REAL (SIDTD) — o juiz definitivo.

## Fase 3 — Features (Fase A/B da memória) 🔄 iniciada
- Fase A: `scripts/features/embeddings_dinov2.py` (embeddings DinoV2) e
  `scripts/treino/treinar_classificador.py` (cabeça leve + avaliação honesta) já existem.
- Falta rodar o treino de fato após os dados estarem prontos (Fase 2).
- Fase B (futura): mapa de NoisePrint/TruFor como segundo canal.

## Fase 4 — Validação de dados (Fase C) ⬜
- Checksum de CPF, MRZ e QR code, independente do sinal de imagem.

## Fase 5 — Treino e avaliação ⬜
- Cabeça de classificação leve sobre embeddings (baseline).
- Validação cruzada leave-one-technique-out (mitigação do "Synthetic Utility Gap").
- Validação externa contra dataset público (SIDTD) para medir o "Reality Gap".

## Fase 6 (opcional) — Privacy-preserving ⬜
- Classificação por patches anonimizados (referência: FakeIDet), se o volume de
  dados/voluntários crescer.
