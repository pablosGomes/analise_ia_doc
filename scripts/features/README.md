# scripts/features/

Extração de features para classificação (Seção 3.3 / 5.3 da documentação
técnica):

- `embeddings_dinov2.py` — embeddings via DinoV2 pré-treinado (Fase A do
  roadmap: baseline rápido, bom mesmo com dataset pequeno).
- `noiseprint.py` — mapa de ruído via TruFor/NoisePrint++ como segundo canal,
  fundido com a imagem RGB (Fase B, inspirado na arquitetura EdgeDoc).

Saída gravada em `datasets/processed/`.
