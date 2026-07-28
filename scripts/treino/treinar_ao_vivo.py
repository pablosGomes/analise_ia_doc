"""Treino da cabeça de classificação por ÉPOCAS, com métricas ao vivo (Fase 3).

MLP leve (torch) sobre os embeddings DinoV2 congelados. Split treino/validação
AGRUPADO por documento (sem vazamento entre variantes). A cada época publica loss
de treino/val e AUC de validação (geral e por técnica) via `Publicador` — que grava
um JSONL local e, se a VPS estiver configurada, envia ao dashboard. No início publica
um evento `estado_gerador` com as contagens do dataset e a foto da sonda (AUC agrupada,
leave-one-technique-out, leave-one-generator-out, separabilidade de fonte).

Uso:
    python -m scripts.treino.treinar_ao_vivo \
        --embeddings datasets/processed/embeddings_dinov2.npz --epocas 120 --intervalo 0.2
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.treino import treinar_classificador as tc
from scripts.treino.publicador import Publicador

SEMENTE = 42


class CabecaMLP(nn.Module):
    def __init__(self, dim_entrada, oculta=256, dropout=0.3):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_entrada, oculta), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(oculta, 1))

    def forward(self, x):
        return self.rede(x).squeeze(-1)


def _split_agrupado(y, grupos, semente=SEMENTE):
    """Um único split treino/val estratificado e AGRUPADO por documento."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
    return next(iter(skf.split(np.zeros(len(y)), y, groups=grupos)))


def _aucs_por_tecnica(y_true, prob, tecnica, fonte, legit_mask):
    """AUC de cada técnica de fraude vs. os legítimos DA MESMA FONTE, na validação —
    restringir a fonte evita a pegadinha de o número refletir 'reais × BID' (0,99)
    em vez da fraude, para técnicas que só existem numa fonte (ex.: troca_foto)."""
    out = {}
    for t in sorted({t for t, r in zip(tecnica, y_true) if r == 1 and t}):
        eh_t = (tecnica == t) & (y_true == 1)
        legit_mesma_fonte = legit_mask & np.isin(fonte, list(set(fonte[eh_t])))
        m = eh_t | legit_mesma_fonte
        if len(set(y_true[m])) == 2:
            out[str(t)] = float(roc_auc_score(y_true[m], prob[m]))
    return out


def _foto_gerador(dados, modelo="logistico"):
    """Snapshot da sonda one-shot para o painel de estado do gerador."""
    X, y = dados["X"], dados["y"]
    tecnica, documento, fonte = dados["tecnica"], dados["documento_origem"], dados["fonte"]
    gerador = dados.get("gerador")
    est = {
        "n_amostras": int(len(y)), "n_fraude": int((y == 1).sum()), "n_legitimo": int((y == 0).sum()),
        "por_tecnica": {str(k): int(v) for k, v in Counter(tecnica[y == 1]).items()},
        "por_gerador": ({str(k): int(v) for k, v in Counter(gerador[y == 1]).items()} if gerador is not None else {}),
        "por_fonte": {str(k): int(v) for k, v in Counter(fonte).items()},
    }
    est["auc_agrupada"] = float(tc.avaliar_split_agrupado(X, y, documento, modelo).mean())
    est["loto"] = {k: float(v) for k, v in tc.avaliar_leave_one_technique_out(X, y, tecnica, fonte, modelo).items()}
    if gerador is not None:
        est["logo"] = {k: float(v) for k, v in tc.avaliar_leave_one_generator_out(X, y, gerador, fonte, modelo).items()}
    sep = tc.avaliar_separabilidade_fonte(X, fonte, documento, modelo)
    est["separabilidade_fonte"] = float(sep.mean()) if sep is not None else None
    return est


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="datasets/processed/embeddings_dinov2.npz")
    parser.add_argument("--epocas", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--oculta", type=int, default=256)
    parser.add_argument("--jsonl", default="datasets/processed/metricas_treino.jsonl")
    parser.add_argument("--intervalo", type=float, default=0.0,
                        help="pausa (s) entre épocas — deixa a curva 'assistível' no dashboard")
    parser.add_argument("--sem-banco", action="store_true",
                        help="não gravar as métricas no MongoDB ao final")
    args = parser.parse_args()

    dados = tc._carregar(Path(args.embeddings))
    X = dados["X"].astype(np.float32)
    y = dados["y"].astype(np.int64)
    tecnica, documento, fonte = dados["tecnica"], dados["documento_origem"], dados["fonte"]

    tr, va = _split_agrupado(y, documento)
    escala = StandardScaler().fit(X[tr])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(escala.transform(X[tr]), dtype=torch.float32, device=dev)
    Xva = torch.tensor(escala.transform(X[va]), dtype=torch.float32, device=dev)
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    yva = y[va]
    yva_t = torch.tensor(yva, dtype=torch.float32, device=dev)

    modelo = CabecaMLP(X.shape[1], args.oculta).to(dev)
    otim = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    perda_fn = nn.BCEWithLogitsLoss()

    pub = Publicador(args.jsonl)
    pub.publicar({"tipo": "inicio", "dispositivo": dev, "epocas": args.epocas,
                  "n_treino": int(len(tr)), "n_val": int(len(va))})
    pub.publicar({"tipo": "estado_gerador", **_foto_gerador(dados)})

    tec_va, fonte_va, legit_va = tecnica[va], fonte[va], (yva == 0)
    for ep in range(1, args.epocas + 1):
        modelo.train(); otim.zero_grad()
        loss = perda_fn(modelo(Xtr), ytr)
        loss.backward(); otim.step()

        modelo.eval()
        with torch.no_grad():
            logit_va = modelo(Xva)
            loss_va = perda_fn(logit_va, yva_t)
            prob_va = torch.sigmoid(logit_va).cpu().numpy()
        auc = float(roc_auc_score(yva, prob_va)) if len(set(yva)) == 2 else float("nan")
        pub.publicar({
            "tipo": "epoca", "epoca": ep,
            "loss_treino": float(loss.item()), "loss_val": float(loss_va.item()),
            "auc_val": auc, "auc_por_tecnica": _aucs_por_tecnica(yva, prob_va, tec_va, fonte_va, legit_va),
        })
        if args.intervalo > 0:
            time.sleep(args.intervalo)
        if ep == 1 or ep % 10 == 0:
            print(f"época {ep:3d} | loss_tr {loss.item():.3f} loss_va {loss_va.item():.3f} auc_va {auc:.3f}")

    pub.publicar({"tipo": "fim", "epocas": args.epocas})
    print("fim — métricas em", args.jsonl)

    if not args.sem_banco:
        # Guarda as métricas desta rodada no MongoDB local (best-effort: se o banco
        # estiver fora do ar, o treino já terminou e o JSONL continua valendo).
        try:
            from scripts.db.ingestao import ingerir_metricas
            from scripts.db.mongo import conectar, garantir_indices
            db = conectar()
            garantir_indices(db)
            n_m, n_a = ingerir_metricas(db, [Path(args.jsonl)])
            print(f"MongoDB: {n_m} épocas e {n_a} avaliação(ões) gravadas")
        except Exception as e:
            print(f"MongoDB indisponível ({e.__class__.__name__}) — métricas ficaram só no JSONL")


if __name__ == "__main__":
    main()
