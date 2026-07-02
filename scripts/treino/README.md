# scripts/treino/

Treino do classificador final sobre os embeddings/features extraídos.
Início recomendado: cabeça de classificação leve (regressão logística ou MLP
pequeno) sobre embeddings DinoV2 (Fase A). Evoluir para fusão com NoisePrint
(Fase B) conforme o dataset crescer.

Sempre validar contra exemplos de fraude gerados por técnica diferente da
usada no treino, para mitigar o "Synthetic Utility Gap" descrito na Seção 7.2
da documentação técnica.
