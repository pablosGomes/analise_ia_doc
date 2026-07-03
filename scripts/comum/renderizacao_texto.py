"""Utilitários para remover texto de uma região (inpaint) e desenhar texto
substituto em seu lugar — usado por `digito_verificador.py` e
`edicao_campos.py` para editar campos de documentos de forma controlada.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CANDIDATOS_FONTE_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_CANDIDATOS_FONTE_DIFERENTE = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]


def _carregar_fonte(candidatos, tamanho_px):
    for caminho in candidatos:
        if Path(caminho).exists():
            try:
                return ImageFont.truetype(caminho, size=max(8, tamanho_px))
            except OSError:
                continue
    return ImageFont.load_default()


def remover_texto_regiao(imagem_bgr, x, y, w, h, margem=3):
    resultado = imagem_bgr.copy()
    alt, larg = resultado.shape[:2]
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)

    mascara = np.zeros((alt, larg), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = 255
    return cv2.inpaint(resultado, mascara, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def estimar_cor_tinta(imagem_bgr, x, y, w, h):
    alt, larg = imagem_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(larg, x + w), min(alt, y + h)
    recorte = imagem_bgr[y0:y1, x0:x1]
    if recorte.size == 0:
        return (30, 30, 30)

    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    limiar = np.percentile(cinza, 15)
    mascara_tinta = cinza <= limiar
    if not mascara_tinta.any():
        return (30, 30, 30)

    cor_mediana = np.median(recorte[mascara_tinta], axis=0)
    return tuple(int(c) for c in cor_mediana)


def _variancia_laplaciana(imagem_bgr):
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


def desenhar_texto(
    imagem_bgr, texto, x, y, w, h,
    fonte_correta=True, cor_bgr=(30, 30, 30), calibrar_nitidez=True,
):
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    draw = ImageDraw.Draw(pil_img)

    tamanho_fonte = max(8, int(h * 0.8))
    candidatos = _CANDIDATOS_FONTE_REGULAR if fonte_correta else _CANDIDATOS_FONTE_DIFERENTE
    fonte = _carregar_fonte(candidatos, tamanho_fonte)

    deslocamento_y = 0 if fonte_correta else max(1, h // 6)
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
    draw.text((x, y + deslocamento_y), texto, font=fonte, fill=cor_rgb)

    resultado = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if calibrar_nitidez:
        alt, larg = resultado.shape[:2]
        margem = max(4, h // 3)
        x0, y0 = max(0, x - margem), max(0, y - margem)
        x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)
        entorno = imagem_bgr[y0:y1, x0:x1]
        patch = resultado[y:y + h, x:x + w]
        if entorno.size and patch.size:
            variancia_alvo = _variancia_laplaciana(entorno)
            if variancia_alvo > 0 and _variancia_laplaciana(patch) > variancia_alvo * 1.3:
                patch_suavizado = patch
                sigma = 0.4
                for _ in range(6):
                    candidato = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
                    patch_suavizado = candidato
                    if _variancia_laplaciana(candidato) <= variancia_alvo * 1.3:
                        break
                    sigma += 0.4
                resultado[y:y + h, x:x + w] = patch_suavizado

    return resultado
