"""Simulador de captura final compartilhado (paridade de pipeline).

É aplicado igualmente aos documentos legítimos e a todas as técnicas de fraude,
com a mesma distribuição aleatória. O objetivo é que a única diferença
sistemática entre a classe legítima e a classe fraude seja a manipulação local
da técnica — nunca uma estatística global (compressão, ruído, blur, geometria),
que um modelo de fundação como o DinoV2 poderia usar como atalho.

Regra de uso: a manipulação específica de cada técnica acontece ANTES; depois a
imagem passa por `simular_captura`; o resultado é salvo em PNG (contêiner sem
perda), de modo que o único artefato de compressão presente é o JPEG de
qualidade/subamostragem ALEATÓRIA aplicado aqui dentro — idêntico em
distribuição para as duas classes.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

from scripts.comum.ruido import ruido_sensor_poisson_gaussiano


def _jitter_geometrico(imagem_bgr, rng):
    altura, largura = imagem_bgr.shape[:2]
    angulo = rng.uniform(-2.5, 2.5)
    escala = rng.uniform(0.94, 1.06)
    M = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, escala)
    rotacionada = cv2.warpAffine(
        imagem_bgr, M, (largura, altura),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    # leve perspectiva (simula ângulo de câmera)
    desloc = rng.uniform(0.0, 0.015) * min(altura, largura)
    origem = np.float32([[0, 0], [largura, 0], [largura, altura], [0, altura]])
    jitter = lambda: rng.uniform(-desloc, desloc)
    destino = np.float32([
        [0 + jitter(), 0 + jitter()], [largura + jitter(), 0 + jitter()],
        [largura + jitter(), altura + jitter()], [0 + jitter(), altura + jitter()],
    ])
    Mp = cv2.getPerspectiveTransform(origem, destino)
    return cv2.warpPerspective(
        rotacionada, Mp, (largura, altura),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    ), {"angulo": round(angulo, 2), "escala": round(escala, 3), "perspectiva_px": round(desloc, 1)}


def _fotometrico(imagem_bgr, rng):
    img = imagem_bgr.astype(np.float32)
    alpha = rng.uniform(0.85, 1.18)          # contraste
    beta = rng.uniform(-22, 22)              # brilho
    ganhos = np.array([rng.uniform(0.92, 1.08) for _ in range(3)], dtype=np.float32)  # balanço de branco
    img = img * alpha * ganhos + beta
    gamma = rng.uniform(0.82, 1.22)
    img = 255.0 * np.clip(img / 255.0, 0, 1) ** gamma
    return img, {"contraste": round(alpha, 3), "brilho": round(beta, 1),
                 "gamma": round(gamma, 3), "balanco_branco": [round(float(g), 3) for g in ganhos]}


def _vinheta(imagem_float, rng):
    if rng.random() > 0.6:
        return imagem_float, 0.0
    altura, largura = imagem_float.shape[:2]
    cx, cy = rng.uniform(0.35, 0.65) * largura, rng.uniform(0.35, 0.65) * altura
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    dist /= dist.max() + 1e-6
    forca = rng.uniform(0.08, 0.22)
    return imagem_float * (1.0 - dist[..., None] * forca), round(forca, 3)


def jpeg_diverso(imagem_bgr, rng):
    """Compressão JPEG com fator de qualidade em faixa larga e subamostragem de
    croma variável, evitando o viés de uma faixa estreita de qualidade. Pode
    aplicar 1 ou 2 gerações. Codifica/decodifica em memória via Pillow para
    controlar a subamostragem."""
    rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    passadas = []
    n = rng.choice([1, 1, 2])
    for _ in range(n):
        import io
        qualidade = rng.randint(35, 96)
        subamostragem = rng.choice([0, 1, 2])  # 4:4:4 / 4:2:2 / 4:2:0
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=qualidade, subsampling=subamostragem)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        passadas.append({"qualidade": qualidade, "subamostragem": subamostragem})
    saida = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return saida, passadas


def simular_captura(imagem_bgr, rng):
    """Aplica o pipeline de captura completo e retorna (imagem_uint8, parametros).
    DEVE ser chamado tanto para legítimos quanto para fraudes, com o mesmo `rng`
    de mesma distribuição."""
    # cap de resolução (realista p/ foto de celular + desempenho); aplicado
    # IGUALMENTE a legítimos e fraudes, preservando a paridade de pipeline.
    MAX_DIM = 1024
    h0, w0 = imagem_bgr.shape[:2]
    escala_cap = 1.0
    if max(h0, w0) > MAX_DIM:
        escala_cap = MAX_DIM / max(h0, w0)
        imagem_bgr = cv2.resize(imagem_bgr, (int(w0 * escala_cap), int(h0 * escala_cap)),
                                interpolation=cv2.INTER_AREA)
    img, p_geo = _jitter_geometrico(imagem_bgr, rng)
    img_f, p_foto = _fotometrico(img, rng)
    img_f, forca_vinheta = _vinheta(img_f, rng)

    ganho_iso = rng.uniform(0.8, 2.6)
    img_f = ruido_sensor_poisson_gaussiano(img_f, ganho_iso, rng)

    img_u = np.clip(img_f, 0, 255).astype(np.uint8)
    sigma_desfoque = 0.0
    if rng.random() < 0.5:
        sigma_desfoque = rng.uniform(0.3, 1.0)
        img_u = cv2.GaussianBlur(img_u, (0, 0), sigmaX=sigma_desfoque)

    img_u, passadas_jpeg = jpeg_diverso(img_u, rng)

    parametros = {
        "geometria": p_geo,
        "fotometrico": p_foto,
        "forca_vinheta": forca_vinheta,
        "ganho_iso_ruido": round(ganho_iso, 3),
        "sigma_desfoque": round(sigma_desfoque, 3),
        "jpeg": passadas_jpeg,
        "escala_cap": round(escala_cap, 4),
    }
    return img_u, parametros
