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


def avaliar_leave_one_technique_out(X, y, tecnica, fonte, tipo_modelo):
    """Para cada técnica de fraude: treina nas demais técnicas + legítimos, testa
    na técnica retida contra os legítimos DA MESMA FONTE (bid/reais).

    Restringir aos legítimos da mesma fonte remove o vazamento de fonte: como
    algumas técnicas só existem numa fonte (ex.: troca_foto só nos 'reais'),
    comparar contra TODOS os legítimos deixaria o modelo separar por "cara de
    documento" (reais × BID, AUC ~0,99) em vez de pela fraude — foi a "pegadinha"
    do 0,978 vista na troca_foto. Legítimos aparecem em treino e teste (não têm
    técnica para reter); no teste servem só de contraste para o cálculo da AUC."""
    tecnicas_fraude = sorted({t for t, r in zip(tecnica, y) if r == 1})
    resultados = {}
    for retida in tecnicas_fraude:
        eh_retida = (tecnica == retida) & (y == 1)
        fontes_da_retida = set(fonte[eh_retida])
        eh_legitimo_mesma_fonte = (y == 0) & np.isin(fonte, list(fontes_da_retida))
        treino_mask = ~eh_retida                        # tudo menos as fraudes retidas
        teste_mask = eh_retida | eh_legitimo_mesma_fonte  # retida vs. legítimos da MESMA fonte
        if len(set(y[teste_mask])) < 2:
            continue
        modelo = _construir_modelo(tipo_modelo)
        modelo.fit(X[treino_mask], y[treino_mask])
        prob = modelo.predict_proba(X[teste_mask])[:, 1]
        resultados[retida] = roc_auc_score(y[teste_mask], prob)
    return resultados


def avaliar_separabilidade_fonte(X, fonte, grupos, tipo_modelo, n_folds=5):
    """Diagnóstico de vazamento de fonte: quão separáveis são as fontes (bid × reais)
    pelos embeddings? AUC alta (~0,9+) significa que misturar fontes num único
    treino/split contamina a métrica — a fonte prediz o rótulo por tabela. Retorna
    None se houver só uma fonte no dataset."""
    fontes = sorted(set(fonte))
    if len(fontes) < 2:
        return None
    y_fonte = (fonte == fontes[0]).astype(int)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE)
    aucs = []
    for treino_idx, teste_idx in skf.split(X, y_fonte, groups=grupos):
        if len(set(y_fonte[treino_idx])) < 2 or len(set(y_fonte[teste_idx])) < 2:
            continue
        modelo = _construir_modelo(tipo_modelo)
        modelo.fit(X[treino_idx], y_fonte[treino_idx])
        aucs.append(roc_auc_score(y_fonte[teste_idx], modelo.predict_proba(X[teste_idx])[:, 1]))
    return np.array(aucs) if aucs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="datasets/processed/embeddings_dinov2.npz")
    parser.add_argument("--modelo", choices=["logistico", "mlp"], default="logistico")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    dados = _carregar(Path(args.embeddings))
    X, y = dados["X"], dados["y"]
    tecnica, documento, fonte = dados["tecnica"], dados["documento_origem"], dados["fonte"]
    print(f"Embeddings: X={X.shape} | fraude={(y == 1).sum()} | legitimo={(y == 0).sum()}")
    print(f"Modelo: {args.modelo}")
    print("-" * 60)

    print("[1] Split AGRUPADO por documento (sem vazamento entre variantes)")
    aucs = avaliar_split_agrupado(X, y, documento, args.modelo, args.folds)
    print(f"    AUC (todas as fontes) = {aucs.mean():.3f} ± {aucs.std():.3f}  (folds: "
          + ", ".join(f"{a:.3f}" for a in aucs) + ")")
    # Também por fonte separadamente: mistura de fontes com proporções fraude/legítimo
    # diferentes infla a AUC agregada (a fonte vira atalho). Ver [diag] abaixo.
    for f in sorted(set(fonte)):
        m = fonte == f
        if len(set(y[m])) == 2:
            a = avaliar_split_agrupado(X[m], y[m], documento[m], args.modelo, args.folds)
            print(f"    AUC (só fonte={f}) = {a.mean():.3f} ± {a.std():.3f}")
    print()

    print("[diag] Separabilidade de fonte (bid × reais) — o quanto o modelo separa a")
    print("       ORIGEM, não a fraude. Alto (~0,9+) => misturar fontes contamina a métrica.")
    sep = avaliar_separabilidade_fonte(X, fonte, documento, args.modelo, args.folds)
    print(f"       AUC de fonte = {sep.mean():.3f} ± {sep.std():.3f}" if sep is not None
          else "       (fonte única — n/a)")
    print()

    print("[2] Leave-one-technique-out (retida vs. legítimos da MESMA fonte)")
    loto = avaliar_leave_one_technique_out(X, y, tecnica, fonte, args.modelo)
    for tec, auc in sorted(loto.items(), key=lambda kv: kv[1]):
        print(f"    {tec:<22} AUC = {auc:.3f}")
    if loto:
        print(f"    média = {np.mean(list(loto.values())):.3f}")
    print()
    print("Interpretação: AUC alta no split agrupado = separa bem legítimo/fraude.")
    print("AUC que cai muito numa técnica retida = modelo depende dessa técnica")
    print("específica (sinal de Synthetic Utility Gap).")


if __name__ == "__main__":
    main()
