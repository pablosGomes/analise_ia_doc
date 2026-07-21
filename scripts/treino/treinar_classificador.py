"""Treino da cabeça de classificação leve sobre embeddings DinoV2 (Fase 3 / Fase A).

Consome o ``.npz`` produzido por ``scripts.features.embeddings_dinov2`` e treina um
classificador leve (regressão logística por padrão) para separar legítimo de fraude.

Dois protocolos de avaliação são reportados, ambos desenhados para não superestimar
o desempenho:

1. Split AGRUPADO por documento de origem. Como várias variantes vêm do mesmo
   documento, um split aleatório vazaria variantes do mesmo documento entre treino
   e teste (data leakage). Aqui todas as variantes de um documento ficam do mesmo
   lado do split.

2. Leave-one-technique-out. Treina com todas as técnicas de fraude menos uma (mais
   os legítimos) e testa na técnica retida. Mede diretamente a generalização para
   um tipo de fraude não visto — o teste que expõe o Synthetic Utility Gap
   (documentacao_deteccao_fraude, Seções 7.2 e 10.2).

Uso:
    python -m scripts.treino.treinar_classificador \
        --embeddings datasets/processed/embeddings_dinov2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42


def _construir_modelo(tipo: str):
    if tipo == "mlp":
        cabeca = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                               early_stopping=True, random_state=SEMENTE)
    else:
        cabeca = LogisticRegression(max_iter=2000, C=1.0)
    return make_pipeline(StandardScaler(), cabeca)


def _carregar(caminho: Path) -> dict:
    dados = np.load(caminho, allow_pickle=True)
    return {k: dados[k] for k in dados.files}


def avaliar_split_agrupado(X, y, grupos, tipo_modelo, n_folds=5):
    """Validação cruzada estratificada com grupos: nenhum documento aparece em
    treino e teste ao mesmo tempo."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE)
    aucs = []
    for treino_idx, teste_idx in skf.split(X, y, groups=grupos):
        modelo = _construir_modelo(tipo_modelo)
        modelo.fit(X[treino_idx], y[treino_idx])
        prob = modelo.predict_proba(X[teste_idx])[:, 1]
        aucs.append(roc_auc_score(y[teste_idx], prob))
    return np.array(aucs)


def avaliar_leave_one_technique_out(X, y, tecnica, tipo_modelo):
    """Para cada técnica de fraude: treina nas demais técnicas + legítimos, testa
    na técnica retida contra os legítimos.

    Nota: legítimos aparecem em treino e teste (não têm "técnica" para reter). O
    que se mede é se o modelo, sem nunca ter visto a técnica retida, ainda a
    reconhece como fraude — por isso os legítimos do teste servem só de contraste
    para o cálculo da AUC, não como novidade avaliada."""
    tecnicas_fraude = sorted({t for t, r in zip(tecnica, y) if r == 1})
    eh_legitimo = (y == 0)
    resultados = {}
    for retida in tecnicas_fraude:
        eh_retida = (tecnica == retida) & (y == 1)
        treino_mask = ~eh_retida            # tudo menos as fraudes da técnica retida
        teste_mask = eh_retida | eh_legitimo  # técnica retida vs. legítimos
        modelo = _construir_modelo(tipo_modelo)
        modelo.fit(X[treino_mask], y[treino_mask])
        prob = modelo.predict_proba(X[teste_mask])[:, 1]
        resultados[retida] = roc_auc_score(y[teste_mask], prob)
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="datasets/processed/embeddings_dinov2.npz")
    parser.add_argument("--modelo", choices=["logistico", "mlp"], default="logistico")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    dados = _carregar(Path(args.embeddings))
    X, y = dados["X"], dados["y"]
    tecnica, documento = dados["tecnica"], dados["documento_origem"]
    print(f"Embeddings: X={X.shape} | fraude={(y == 1).sum()} | legitimo={(y == 0).sum()}")
    print(f"Modelo: {args.modelo}")
    print("-" * 60)

    print("[1] Split AGRUPADO por documento (sem vazamento entre variantes)")
    aucs = avaliar_split_agrupado(X, y, documento, args.modelo, args.folds)
    print(f"    AUC = {aucs.mean():.3f} ± {aucs.std():.3f}  (folds: "
          + ", ".join(f"{a:.3f}" for a in aucs) + ")")
    print()

    print("[2] Leave-one-technique-out (treina nas demais técnicas + legítimos)")
    loto = avaliar_leave_one_technique_out(X, y, tecnica, args.modelo)
    for tec, auc in sorted(loto.items(), key=lambda kv: kv[1]):
        print(f"    {tec:<22} AUC = {auc:.3f}")
    print(f"    média = {np.mean(list(loto.values())):.3f}")
    print()
    print("Interpretação: AUC alta no split agrupado = separa bem legítimo/fraude.")
    print("AUC que cai muito numa técnica retida = modelo depende dessa técnica")
    print("específica (sinal de Synthetic Utility Gap).")


if __name__ == "__main__":
    main()
