"""Harmonização de textura de regiões editadas.

Uma região manipulada (preenchida por inpainting, ou com texto redesenhado em cor
sólida) fica com MENOS micro-textura de alta frequência do que o documento ao
redor: falta o grão de papel/impressão, o retículo, o guilhoché e o histórico de
compressão que o resto da imagem carrega. Como toda a imagem passa depois pelo
MESMO simulador de captura (mesmo ruído para as duas classes), essa diferença
relativa de textura NÃO é apagada — ela sobrevive, e um modelo de fundação como o
DinoV2 a usa como "assinatura de edição", separando fraude de legítimo sem olhar a
manipulação em si.

Este módulo transplanta a textura de alta frequência de um vizinho REAL do próprio
documento para dentro da região editada, de forma ADITIVA: preserva a estrutura da
edição (o dígito trocado, o texto novo, a região movida) e apenas repõe o grão que
faltava, aproximando as estatísticas locais da região das do entorno.
"""

from __future__ import annotations

import cv2
import numpy as np


def _residual_alta_freq(patch_bgr, sigma):
    """Componente de alta frequência (textura) do patch: original menos sua
    versão suavizada. Média ~zero."""
    base = cv2.GaussianBlur(patch_bgr.astype(np.float32), (0, 0), sigmaX=sigma)
    return patch_bgr.astype(np.float32) - base


def _mapa_std_local(residual, sigma_suave):
    """Mapa (h, w) do desvio-padrão LOCAL da textura, calculado por pixel a partir
    da energia do residual suavizada. Alto onde há textura/borda; ~0 onde é liso."""
    energia = residual.astype(np.float32) ** 2
    if energia.ndim == 3:
        energia = energia.mean(axis=2)
    return np.sqrt(cv2.GaussianBlur(energia, (0, 0), sigmaX=sigma_suave) + 1e-6)


def _energia_borda(patch_bgr):
    cinza = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


def _doador_vizinho(imagem_bgr, x, y, w, h):
    """Escolhe um patch (w, h) vizinho — dentro da imagem, sem sobrepor a região —
    com a MENOR energia de borda (o mais uniforme, i.e. fundo/grão, sem glifos ou
    linhas fortes que virariam 'fantasmas' ao serem transplantados)."""
    alt, larg = imagem_bgr.shape[:2]
    candidatos = []
    for dx, dy in [(-w, 0), (w, 0), (0, -h), (0, h), (-w, -h), (w, h), (w, -h), (-w, h)]:
        cx, cy = x + dx, y + dy
        if 0 <= cx <= larg - w and 0 <= cy <= alt - h:
            sub = imagem_bgr[cy:cy + h, cx:cx + w]
            if sub.shape[:2] == (h, w):
                candidatos.append(sub)
    if not candidatos:
        return None
    return min(candidatos, key=_energia_borda)


def harmonizar_textura(imagem_bgr, x, y, w, h, rng, forca=None):
    """Injeta, de forma aditiva, a textura de alta frequência de um vizinho real na
    região (x, y, w, h) da imagem (modificada in place e também retornada).

    Não destrói a manipulação: apenas soma o grão que falta. Se não houver vizinho
    válido ou a região for minúscula, retorna a imagem inalterada."""
    alt, larg = imagem_bgr.shape[:2]
    x = int(max(0, min(x, larg - 1))); y = int(max(0, min(y, alt - 1)))
    w = int(max(1, min(w, larg - x))); h = int(max(1, min(h, alt - y)))
    if w < 6 or h < 6:
        return imagem_bgr

    doador = _doador_vizinho(imagem_bgr, x, y, w, h)
    if doador is None:
        return imagem_bgr

    sigma = rng.uniform(1.2, 2.2)
    sigma_suave = max(2.0, min(w, h) / 6.0)
    grao = _residual_alta_freq(doador, sigma)
    # Espelha o grão do doador (decorrelaciona qualquer estrutura remanescente).
    if rng.random() < 0.5:
        grao = grao[::-1]
    if rng.random() < 0.5:
        grao = grao[:, ::-1]

    regiao = imagem_bgr[y:y + h, x:x + w].astype(np.float32)
    alvo = float(np.median(_mapa_std_local(grao, sigma_suave))) + 1e-6      # textura-alvo (do vizinho real)
    std_local_regiao = _mapa_std_local(_residual_alta_freq(regiao, sigma), sigma_suave)

    # Déficit POR PIXEL: quanto de textura falta em cada ponto. Alto nas áreas
    # lisas (entre glifos, buraco de inpaint), ~0 nas bordas de glifo que já têm
    # alta frequência — assim o grão preenche o liso sem borrar a edição.
    deficit = np.clip(alvo - std_local_regiao, 0.0, alvo) / alvo

    forca = forca if forca is not None else rng.uniform(0.7, 1.0)
    nova = regiao + grao * deficit[..., None] * forca
    imagem_bgr[y:y + h, x:x + w] = np.clip(nova, 0, 255).astype(np.uint8)
    return imagem_bgr
