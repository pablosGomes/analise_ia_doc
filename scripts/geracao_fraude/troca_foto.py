"""Técnica — Substituição da foto de rosto (face swap).

Cola o rosto de outro documento sobre o rosto do documento alvo. A versão sutil
ALINHA o rosto novo à posição/escala/rotação do rosto alvo usando a malha de
landmarks (`scripts/comum/rosto.py`, MediaPipe), recorta pela silhueta do rosto
(não por um retângulo), harmoniza cor e nitidez e faz o blend de Poisson — um
salto sobre a colagem por caixa. A versão evidente é a colagem retangular simples
(par fácil × difícil). O restante da variação vem do simulador de captura
compartilhado, aplicado também aos legítimos.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.comum import rosto
from scripts.geracao_fraude import saida
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

# Landmarks estáveis (cantos dos olhos, nariz, boca, queixo, testa) para estimar
# o alinhamento entre o rosto fonte e o rosto alvo.
_PONTOS_ALINHAMENTO = [33, 263, 133, 362, 168, 1, 61, 291, 0, 17, 152, 10]


def _variancia_laplaciana(imagem_bgr):
    return float(cv2.Laplacian(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _mascara_rosto(landmarks, shape, encolher=0.90):
    """Máscara uint8 (0/255) da silhueta do rosto (fecho convexo dos landmarks),
    levemente encolhida para o blend cair dentro da pele, não na borda."""
    pts = landmarks.astype(np.int32)
    centro = pts.mean(axis=0)
    pts = (centro + (pts - centro) * encolher).astype(np.int32)
    mascara = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mascara, cv2.convexHull(pts), 255)
    return mascara


def _harmonizar_cor(warped_bgr, alvo_bgr, mascara_bool):
    """Transferência de cor Reinhard (LAB) usando as estatísticas do rosto ALVO
    dentro da máscara — casa o tom de pele do rosto colado ao do documento."""
    lab_w = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(alvo_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    amostra_w = lab_w[mascara_bool]
    amostra_a = lab_a[mascara_bool]
    if amostra_w.size == 0 or amostra_a.size == 0:
        return warped_bgr
    media_w, desvio_w = amostra_w.mean(0), amostra_w.std(0) + 1e-6
    media_a, desvio_a = amostra_a.mean(0), amostra_a.std(0) + 1e-6
    lab_w = (lab_w - media_w) * (desvio_a / desvio_w) + media_a
    return cv2.cvtColor(np.clip(lab_w, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _igualar_nitidez(warped_bgr, mascara_bool, variancia_alvo, tentativas=6):
    """Suaviza o rosto colado até a nitidez casar com a do alvo (o rosto colado
    costuma vir mais nítido que a foto do documento)."""
    atual = _variancia_laplaciana(warped_bgr)
    if atual <= variancia_alvo * 1.2:
        return warped_bgr
    resultado = warped_bgr
    sigma = 0.4
    for _ in range(tentativas):
        resultado = cv2.GaussianBlur(warped_bgr, (0, 0), sigmaX=sigma)
        if _variancia_laplaciana(resultado) <= variancia_alvo * 1.2:
            break
        sigma += 0.4
    return resultado


def aplicar_swap_alinhado(rosto_alvo, rosto_fonte):
    """Alinha o rosto fonte ao alvo por landmarks, harmoniza e faz o blend de
    Poisson na imagem do alvo (já na rotação em que o rosto ficou em pé).
    Retorna (imagem_bgr, params) ou None."""
    alvo = rosto_alvo.imagem
    src_pts = rosto_fonte.landmarks[_PONTOS_ALINHAMENTO]
    dst_pts = rosto_alvo.landmarks[_PONTOS_ALINHAMENTO]
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return None

    warped = cv2.warpAffine(rosto_fonte.imagem, M, (alvo.shape[1], alvo.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mascara = _mascara_rosto(rosto_alvo.landmarks, alvo.shape)
    mascara_bool = mascara > 0
    if not mascara_bool.any():
        return None

    warped = _harmonizar_cor(warped, alvo, mascara_bool)
    # Nitidez de referência medida na CAIXA da máscara (região 2D real do rosto
    # alvo). Indexar a máscara direto (alvo[mascara_bool]) devolve os pixels em
    # ordem raster — não são vizinhos espaciais, e o Laplaciano nisso é ruído.
    if mascara_bool.sum() > 8:
        ys_m, xs_m = np.where(mascara_bool)
        recorte_alvo = alvo[ys_m.min():ys_m.max() + 1, xs_m.min():xs_m.max() + 1]
        ref = _variancia_laplaciana(recorte_alvo)
    else:
        ref = _variancia_laplaciana(alvo)
    warped = _igualar_nitidez(warped, mascara_bool, ref)

    xs = rosto_alvo.landmarks[:, 0]; ys = rosto_alvo.landmarks[:, 1]
    centro = (int(xs.mean()), int(ys.mean()))
    try:
        composto = cv2.seamlessClone(warped, alvo, mascara, centro, cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: blend alpha com borda suave.
        m = cv2.GaussianBlur(mascara.astype(np.float32) / 255.0, (0, 0), sigmaX=5)[..., None]
        composto = np.clip(warped.astype(np.float32) * m + alvo.astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)

    if rosto_alvo.codigo_rotacao_inversa is not None:
        composto = cv2.rotate(composto, rosto_alvo.codigo_rotacao_inversa)
    return composto, {"modo": "alinhado", "confianca_alvo": round(rosto_alvo.confianca, 3),
                      "confianca_fonte": round(rosto_fonte.confianca, 3)}


def _clampar_caixa(caixa, shape):
    """Clampa (x, y, w, h) aos limites da imagem (origem >= 0, sem extrapolar).
    A caixa do MediaPipe pode ter origem negativa ou passar da borda."""
    x, y, w, h = caixa
    alt, larg = shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(larg, x + w), min(alt, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def aplicar_colagem_simples(rosto_alvo, rosto_fonte):
    """Colagem retangular direta do rosto fonte na caixa do rosto alvo (variante
    evidente). Retorna (imagem_bgr, params) na orientação original do alvo."""
    alvo = rosto_alvo.imagem.copy()
    ax, ay, aw, ah = _clampar_caixa(rosto_alvo.caixa, alvo.shape)
    fx, fy, fw, fh = _clampar_caixa(rosto_fonte.caixa, rosto_fonte.imagem.shape)
    if min(aw, ah, fw, fh) <= 0:
        return None
    recorte = rosto_fonte.imagem[fy:fy + fh, fx:fx + fw]
    if recorte.size == 0:
        return None
    alvo[ay:ay + ah, ax:ax + aw] = cv2.resize(recorte, (aw, ah))
    if rosto_alvo.codigo_rotacao_inversa is not None:
        alvo = cv2.rotate(alvo, rosto_alvo.codigo_rotacao_inversa)
    return alvo, {"modo": "colagem_simples", "caixa_alvo": [ax, ay, aw, ah]}


_MODOS = [("alinhado", aplicar_swap_alinhado, "sutil"),
          ("colagem_simples", aplicar_colagem_simples, "evidente")]


def gerar_variantes_para_par(caminho_alvo, caminho_fonte, tipo_documento):
    alvo = cv2.imread(str(caminho_alvo)); fonte = cv2.imread(str(caminho_fonte))
    if alvo is None or fonte is None:
        return []
    rosto_alvo = rosto.localizar(alvo)
    rosto_fonte = rosto.localizar(fonte)
    if rosto_alvo is None or rosto_fonte is None:
        return []
    saidas = []
    for nome_modo, funcao, dificuldade in _MODOS:
        res = funcao(rosto_alvo, rosto_fonte)
        if res is None:
            continue
        img, params = res
        params["fonte_rosto"] = str(caminho_fonte)
        reg = RegistroFraude(
            tecnica="troca_foto", documento_origem=str(caminho_alvo), tipo_documento=tipo_documento,
            arquivo_gerado="", campo_alterado="foto_rosto", dificuldade=dificuldade,
            rotulo="fraude", parametros=params, gerador="landmark_classico",
        )
        saidas.append((img, reg, nome_modo))
    return saidas


def processar_dataset(pasta_legitimos, pasta_saida, variantes_por_documento, semente=42):
    rng = random.Random(semente)
    arquivos_por_tipo = {}
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        arqs = sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM)
        if arqs:
            arquivos_por_tipo[pasta_tipo.name] = arqs

    pasta_saida_tecnica = pasta_saida / "troca_foto"
    total = 0
    for tipo_documento, arquivos in arquivos_por_tipo.items():
        for caminho_alvo in arquivos:
            candidatos = [a for a in arquivos if a != caminho_alvo] or \
                         [a for l in arquivos_por_tipo.values() for a in l if a != caminho_alvo]
            if not candidatos:
                continue
            # Usa o rng da sessão (não um novo Random(semente) por alvo, que daria
            # a MESMA ordem de fontes para todos os documentos) — assim cada alvo
            # embaralha suas fontes de forma diferente, diversificando os pares.
            rng.shuffle(candidatos)
            pares = 0
            for caminho_fonte in itertools.cycle(candidatos):
                if pares * len(_MODOS) >= variantes_por_documento:
                    break
                for img, reg, modo in gerar_variantes_para_par(caminho_alvo, caminho_fonte, tipo_documento):
                    stem = f"{caminho_alvo.stem}__troca_foto_{modo}_{pares}"
                    saida.finalizar(img, rng, pasta_saida_tecnica, stem, reg)
                    total += 1
                pares += 1
                if len(candidatos) == 1 and pares > 20:
                    break
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/gerado/reais_fraude")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()
    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"troca_foto: {total} imagem(ns) gerada(s)")


if __name__ == "__main__":
    main()
