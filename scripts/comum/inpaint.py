"""Remoção do texto original de um campo, preservando o fundo do documento.

Apagar a caixa inteira de um campo deixa um borrão retangular — uma marca visível
e um atalho para o classificador. Aqui removemos APENAS os traços de tinta do
texto, mantendo intacto o padrão de fundo (guilhoché, tinta de segurança) entre e
ao redor das letras. O método de inpaint dos traços é sorteado (Telea/NS) para não
deixar uma assinatura de inpainting constante.
"""

from __future__ import annotations

import cv2
import numpy as np


def remover_tinta(imagem_bgr, x, y, w, h, rng, margem=3):
    """Remove apenas os traços de tinta do texto na região (x, y, w, h),
    preservando o padrão de fundo. Retorna (imagem, nome_do_metodo)."""
    alt, larg = imagem_bgr.shape[:2]
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)
    recorte = imagem_bgr[y0:y1, x0:x1]
    if recorte.size == 0:
        return imagem_bgr.copy(), "tinta_vazio"

    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    p15, med, p85 = np.percentile(cinza, [15, 50, 85])
    # Polaridade: texto escuro sobre fundo claro, ou claro sobre escuro (CPF digital).
    texto_escuro = (med - p15) >= (p85 - med)
    if texto_escuro:
        _, mascara_tinta = cv2.threshold(cinza, int((med + p15) / 2), 255, cv2.THRESH_BINARY_INV)
    else:
        _, mascara_tinta = cv2.threshold(cinza, int((med + p85) / 2), 255, cv2.THRESH_BINARY)
    # Dilata levemente para cobrir o anti-aliasing das bordas dos glifos.
    mascara_tinta = cv2.dilate(mascara_tinta, np.ones((3, 3), np.uint8), iterations=1)

    metodo = rng.choice(("telea", "ns"))
    flag = cv2.INPAINT_TELEA if metodo == "telea" else cv2.INPAINT_NS
    # Raio sorteado: um raio fixo deixa uma assinatura de inpaint constante que o
    # DinoV2 poderia decorar; variar dilui essa marca de ferramenta.
    raio = rng.randint(2, 4)
    mascara = np.zeros((alt, larg), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = mascara_tinta
    return cv2.inpaint(imagem_bgr, mascara, inpaintRadius=raio, flags=flag), "tinta_" + metodo
