"""Utilitários para remover texto de uma região (inpaint) e desenhar texto
substituto em seu lugar — usado por `digito_verificador.py` e
`edicao_campos.py` para editar campos de documentos de forma controlada.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Caminhos comuns de fontes TrueType em distribuições Linux (VPS Ubuntu/Debian).
# Se nenhuma existir, cai para a fonte bitmap padrão do Pillow (pior
# qualidade visual, mas nunca quebra a execução).
_CANDIDATOS_FONTE_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_CANDIDATOS_FONTE_DIFERENTE = [
    # Fonte deliberadamente diferente da usada nos documentos reais (que tende
    # a ser um monoespaçado tipo OCR-B) — para a variante "fonte_incorreta".
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]


def _carregar_fonte(candidatos: list[str], tamanho_px: int) -> ImageFont.ImageFont:
    for caminho in candidatos:
        if Path(caminho).exists():
            try:
                return ImageFont.truetype(caminho, size=max(8, tamanho_px))
            except OSError:
                continue
    return ImageFont.load_default()


def remover_texto_regiao(imagem_bgr: np.ndarray, x: int, y: int, w: int, h: int, margem: int = 3) -> np.ndarray:
    """Remove o texto original da região via inpaint (Telea) — preserva
    melhor a textura de fundo do documento do que um preenchimento sólido,
    reduzindo o risco de o próprio patch virar um sinal forense óbvio."""
    resultado = imagem_bgr.copy()
    alt, larg = resultado.shape[:2]
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)

    mascara = np.zeros((alt, larg), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = 255
    return cv2.inpaint(resultado, mascara, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def desenhar_texto(
    imagem_bgr: np.ndarray,
    texto: str,
    x: int,
    y: int,
    w: int,
    h: int,
    fonte_correta: bool = True,
    cor_bgr: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """Desenha `texto` dentro da caixa (x, y, w, h). Se `fonte_correta` for
    False, usa uma fonte propositalmente diferente (itálico/serifado) e um
    leve desalinhamento vertical, simulando uma fraude malfeita — o
    classificador deve aprender ambos os casos, não só o óbvio."""
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    draw = ImageDraw.Draw(pil_img)

    tamanho_fonte = max(8, int(h * 0.8))
    candidatos = _CANDIDATOS_FONTE_REGULAR if fonte_correta else _CANDIDATOS_FONTE_DIFERENTE
    fonte = _carregar_fonte(candidatos, tamanho_fonte)

    deslocamento_y = 0 if fonte_correta else max(1, h // 6)
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
    draw.text((x, y + deslocamento_y), texto, font=fonte, fill=cor_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
