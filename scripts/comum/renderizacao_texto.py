"""Utilitários para remover texto de uma região e desenhar texto substituto,
usados por `edicao_campos.py` (reais) e `gerar_dataset_bid.py` (BID). A fonte pode
ser passada explicitamente (pool multiplataforma em scripts/comum/valores.py) e a
remoção do texto original é feita pelos métodos de scripts/comum/inpaint.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.comum import valores
from scripts.comum.textura import harmonizar_textura


def _carregar_fonte(candidatos, tamanho_px):
    for caminho in candidatos:
        if caminho and Path(caminho).exists():
            try:
                return ImageFont.truetype(caminho, size=max(8, tamanho_px))
            except OSError:
                continue
    return ImageFont.load_default()


def remover_texto_regiao(imagem_bgr, x, y, w, h, margem=3):
    """Remoção simples por inpainting TELEA. Para diversidade de métodos, prefira scripts.comum.inpaint."""
    resultado = imagem_bgr.copy()
    alt, larg = resultado.shape[:2]
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)
    mascara = np.zeros((alt, larg), dtype=np.uint8)
    mascara[y0:y1, x0:x1] = 255
    return cv2.inpaint(resultado, mascara, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def estimar_cor_tinta(imagem_bgr, x, y, w, h):
    """Estima a cor da 'tinta' (texto) numa região, CIENTE DE POLARIDADE: funciona
    tanto para texto escuro sobre fundo claro (papel comum) quanto para texto
    claro sobre fundo escuro (ex.: CPF/CNH digital, comuns no BID). O texto é a
    minoria de pixels mais separada da mediana (o fundo); a polaridade decide se
    a tinta são os pixels mais escuros ou os mais claros."""
    alt, larg = imagem_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(larg, x + w), min(alt, y + h)
    recorte = imagem_bgr[y0:y1, x0:x1]
    if recorte.size == 0:
        return (30, 30, 30)
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    p15, mediana_lum, p85 = np.percentile(cinza, [15, 50, 85])
    texto_escuro = (mediana_lum - p15) >= (p85 - mediana_lum)
    mascara_tinta = cinza <= p15 if texto_escuro else cinza >= p85
    if not mascara_tinta.any():
        return (30, 30, 30) if texto_escuro else (225, 225, 225)
    cor = np.median(recorte[mascara_tinta], axis=0)
    # Garante CONTRASTE MÍNIMO com o fundo: quando o campo original já é de baixo
    # contraste, a cor estimada sairia fraca e o texto reescrito ficaria ilegível
    # — uma "fraude" visualmente idêntica ao legítimo (amostra poluída). Se o
    # contraste de luminância for baixo, força a tinta ao extremo da polaridade.
    lum_tinta = float(0.114 * cor[0] + 0.587 * cor[1] + 0.299 * cor[2])
    if abs(lum_tinta - float(mediana_lum)) < 40:
        return (25, 25, 25) if texto_escuro else (230, 230, 230)
    return tuple(int(c) for c in cor)


def _variancia_laplaciana(imagem_bgr):
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


def extensao_texto(imagem_bgr, x, y, w, h, max_expandir=1.0):
    """Expande a caixa HORIZONTALMENTE para cobrir toda a tinta contígua do texto
    original. A caixa vinda do OCR/gt às vezes corta parte do texto; se a remoção
    e o redesenho usarem só essa caixa curta, sobra um "fantasma" do texto original
    ao lado do novo — um atalho visível. Aqui a caixa é alargada até encontrar uma
    coluna vazia (gap real entre campos), sem invadir o campo vizinho.

    Retorna (x, y, w, h) possivelmente mais largo (nunca mais estreito)."""
    alt, larg = imagem_bgr.shape[:2]
    y0, y1 = max(0, y), min(alt, y + h)
    ex = int(w * max_expandir)
    sx0, sx1 = max(0, x - ex), min(larg, x + w + ex)
    faixa = imagem_bgr[y0:y1, sx0:sx1]
    if faixa.size == 0:
        return (x, y, w, h)
    cinza = cv2.cvtColor(faixa, cv2.COLOR_BGR2GRAY)
    p15, med, p85 = np.percentile(cinza, [15, 50, 85])
    texto_escuro = (med - p15) >= (p85 - med)
    if texto_escuro:
        mascara = cinza <= (p15 + (med - p15) * 0.5)
    else:
        mascara = cinza >= (p85 - (p85 - med) * 0.5)
    coluna_tem_tinta = mascara.sum(axis=0).astype(float) > max(1.0, 0.08 * mascara.shape[0])
    if not coluna_tem_tinta.any():
        return (x, y, w, h)
    # Agrupa colunas com tinta tolerando pequenos vãos (espaços entre letras/palavras).
    gap_tol = max(2, int(0.6 * h))
    indices = np.where(coluna_tem_tinta)[0]
    grupos = []
    ini = prev = indices[0]
    for i in indices[1:]:
        if i - prev <= gap_tol:
            prev = i
        else:
            grupos.append((ini, prev)); ini = prev = i
    grupos.append((ini, prev))
    # Escolhe o grupo que contém o centro da caixa (ou o mais próximo).
    centro_rel = (x + w // 2) - sx0
    escolhido = next((g for g in grupos if g[0] <= centro_rel <= g[1]), None)
    if escolhido is None:
        escolhido = min(grupos, key=lambda g: abs((g[0] + g[1]) // 2 - centro_rel))
    nx0, nx1 = sx0 + escolhido[0], sx0 + escolhido[1] + 1
    nx0, nx1 = min(nx0, x), max(nx1, x + w)  # nunca encolher além da caixa original
    return (nx0, y, nx1 - nx0, h)


def nitidez_regiao(imagem_bgr, x, y, w, h):
    """Nitidez (variância do Laplaciano) de uma região. Usada para medir a nitidez
    do TEXTO ORIGINAL antes de removê-lo, e assim redesenhar o texto novo com a
    mesma nitidez — evitando que a edição fique mais mole/borrada que o resto do
    documento (um atalho visível para o classificador)."""
    alt, larg = imagem_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(larg, x + w), min(alt, y + h)
    recorte = imagem_bgr[y0:y1, x0:x1]
    return _variancia_laplaciana(recorte) if recorte.size else 0.0


def desenhar_texto(
    imagem_bgr, texto, x, y, w, h,
    fonte_correta=True, cor_bgr=(30, 30, 30), calibrar_nitidez=True,
    caminho_fonte=None, rng=None, harmonizar=True, nitidez_alvo=None,
):
    """Desenha `texto` na região. Se `caminho_fonte` for dado, usa essa fonte;
    senão cai nos candidatos padrão. `fonte_correta=False` sem caminho usa uma
    família diferente, mas plausível (não itálico gritante).

    Com `harmonizar=True` e um `rng`, a região editada recebe, ao final, a
    micro-textura de um vizinho real do documento (scripts.comum.textura), para o
    texto redesenhado não ficar liso/chapado demais e virar um atalho para o
    classificador."""
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    draw = ImageDraw.Draw(pil_img)

    candidatos = ([caminho_fonte] if caminho_fonte
                  else (valores.FONTES_REGULARES if fonte_correta else valores.FONTES_SUTIS))

    # Dimensiona a fonte para o texto PREENCHER A LARGURA da caixa, como o texto
    # original preenchia o campo, com teto de altura. Antes dimensionava pela altura
    # e só encolhia para caber na largura — o que deixava textos longos (CPF
    # formatado, nomes) pequenos, sem cobrir o campo e revelando o texto antigo.
    alvo = max(8, int(h * 1.4))
    fonte = _carregar_fonte(candidatos, alvo)
    bbox = fonte.getbbox(texto)
    tw = bbox[2] - bbox[0]
    if tw > 0:
        alvo = max(8, int(alvo * (w * 0.98) / tw))
        fonte = _carregar_fonte(candidatos, alvo)
        bbox = fonte.getbbox(texto)
    th = bbox[3] - bbox[1]
    if th > h * 1.3 and th > 0:            # teto de altura: não estourar verticalmente
        alvo = max(8, int(alvo * (h * 1.3) / th))
        fonte = _carregar_fonte(candidatos, alvo)
        bbox = fonte.getbbox(texto)
        th = bbox[3] - bbox[1]

    # Centraliza na caixa, corrigindo o offset do glifo (bbox[0]/bbox[1]).
    tw = bbox[2] - bbox[0]
    px = x + max(0, (w - tw) // 2) - bbox[0]
    py = y + max(0, (h - th) // 2) - bbox[1]
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
    draw.text((px, py), texto, font=fonte, fill=cor_rgb)
    resultado = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if calibrar_nitidez:
        alt, larg = resultado.shape[:2]
        margem = max(4, h // 3)
        x0, y0 = max(0, x - margem), max(0, y - margem)
        x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)
        entorno = imagem_bgr[y0:y1, x0:x1]
        patch = resultado[y:y + h, x:x + w]
        if entorno.size and patch.size:
            # Alvo de nitidez: idealmente a nitidez do TEXTO ORIGINAL que foi
            # substituído (nitidez_alvo, medido antes do inpaint) — assim o texto
            # novo fica tão nítido quanto o que estava ali, nem mais nem menos.
            # Sem esse valor, cai no entorno; mas o entorno costuma estar liso pela
            # remoção recente, o que borrava demais o texto novo e o deixava mole
            # (um atalho visível). O piso evita destruir a edição de qualquer forma.
            if nitidez_alvo is not None:
                variancia_alvo = max(float(nitidez_alvo), 120.0)
            else:
                variancia_alvo = max(_variancia_laplaciana(entorno), 120.0)
            # TETO no blur: no máx 3 passadas (sigma até ~1.2), nunca apagar.
            if _variancia_laplaciana(patch) > variancia_alvo * 1.3:
                sigma = 0.4
                for _ in range(3):
                    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
                    if _variancia_laplaciana(patch) <= variancia_alvo * 1.3:
                        break
                    sigma += 0.4
                resultado[y:y + h, x:x + w] = patch

    if harmonizar and rng is not None:
        # Injeta a textura de fundo que falta na região reescrita, para as
        # estatísticas locais baterem com o entorno (anti-atalho).
        margem = max(2, h // 6)
        resultado = harmonizar_textura(
            resultado, x - margem, y - margem, w + 2 * margem, h + 2 * margem, rng)
    return resultado
