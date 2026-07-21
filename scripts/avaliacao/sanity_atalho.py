"""Sonda anti-atalho.

Treina um classificador linear usando apenas estatísticas globais de cada imagem
(sem olhar a região manipulada): média/desvio por canal, variância do Laplaciano,
energia de alta frequência (FFT), blocagem 8x8 (JPEG) e estimativa de ruído. Se
esse classificador "somente-global" separar legítimos de fraudes com AUC alta,
existe um atalho global que o modelo de imagem poderia explorar em vez da
manipulação em si; o objetivo é manter essa AUC perto do acaso (≈0.5-0.6). Também
roda o leave-one-technique-out.

Uso:
    python3 -m scripts.avaliacao.sanity_atalho \
        --fraude datasets/gerado/reais_fraude --legitimos datasets/gerado/reais_legitimos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXT = (".png", ".jpg", ".jpeg")


def _blocagem_8x8(cinza):
    """Diferença média nas fronteiras de bloco 8x8 vs. dentro do bloco — proxy
    de artefato de compressão JPEG."""
    dv = np.abs(np.diff(cinza.astype(np.float32), axis=0))
    dh = np.abs(np.diff(cinza.astype(np.float32), axis=1))
    bordas_v = dv[7::8].mean() if dv[7::8].size else 0.0
    bordas_h = dh[:, 7::8].mean() if dh[:, 7::8].size else 0.0
    interno_v = dv.mean() + 1e-6
    interno_h = dh.mean() + 1e-6
    return (bordas_v / interno_v + bordas_h / interno_h) / 2.0


def features_globais(imagem_bgr):
    img = cv2.resize(imagem_bgr, (256, 256))
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = []
    for c in range(3):
        f += [img[..., c].mean(), img[..., c].std()]
    f.append(float(cv2.Laplacian(cinza, cv2.CV_64F).var()))
    # energia de alta frequência (FFT)
    F = np.fft.fftshift(np.fft.fft2(cinza.astype(np.float32)))
    mag = np.abs(F)
    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    alta = mag[((x - cx) ** 2 + (y - cy) ** 2) > 60 ** 2].sum()
    f.append(float(alta / (mag.sum() + 1e-6)))
    f.append(float(_blocagem_8x8(cinza)))
    # estimativa de ruído (mediana do abs do passa-alta)
    hp = cinza.astype(np.float32) - cv2.GaussianBlur(cinza.astype(np.float32), (0, 0), 1.0)
    f.append(float(np.median(np.abs(hp))))
    return np.array(f, dtype=np.float32)


def _carregar(pasta, rotulo, tecnica_de_pasta=False):
    X, y, tec = [], [], []
    for p in Path(pasta).rglob("*"):
        if p.suffix.lower() not in EXT:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        X.append(features_globais(img)); y.append(rotulo)
        tec.append(p.parent.name if tecnica_de_pasta else "legitimo")
    return X, y, tec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraude", default="datasets/gerado/reais_fraude")
    parser.add_argument("--legitimos", default="datasets/gerado/reais_legitimos")
    args = parser.parse_args()

    Xf, yf, tf = _carregar(args.fraude, 1, tecnica_de_pasta=True)
    Xl, yl, tl = _carregar(args.legitimos, 0)
    X = np.array(Xf + Xl); y = np.array(yf + yl); tec = np.array(tf + tl)
    print(f"Amostras: fraude={len(Xf)} | legitimo={len(Xl)} | total={len(X)}")
    if len(set(y)) < 2 or len(X) < 10:
        raise SystemExit("Amostras insuficientes para o teste.")

    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    auc = cross_val_score(modelo, X, y, cv=min(5, len(Xf), len(Xl)), scoring="roc_auc")
    print(f"\n[Sonda somente-global] AUC = {auc.mean():.3f} ± {auc.std():.3f}")
    print("  Interpretação: ~0.5-0.6 = SEM atalho global (bom). Alto = atalho global presente.")

    # leave-one-technique-out
    tecnicas = sorted(set(tf))
    print("\n[Leave-one-technique-out] (treina nas demais técnicas + legítimos, testa na retida)")
    for held in tecnicas:
        mask_test = (tec == held)
        mask_train = ~mask_test & ((y == 0) | (tec != held))
        # treino: legítimos + fraudes das outras técnicas; teste: fraudes da técnica retida vs legítimos
        Xtr = X[(tec != held)]; ytr = y[(tec != held)]
        Xte = np.vstack([X[tec == held], X[y == 0]]); yte = np.concatenate([y[tec == held], y[y == 0]])
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            print(f"  - {held}: (dados insuficientes)"); continue
        modelo.fit(Xtr, ytr)
        from sklearn.metrics import roc_auc_score
        prob = modelo.predict_proba(Xte)[:, 1]
        print(f"  - {held}: AUC global-only = {roc_auc_score(yte, prob):.3f}")


if __name__ == "__main__":
    main()
