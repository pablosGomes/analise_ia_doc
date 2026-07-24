"""Transplante de glifos reais — edição de texto usando a fonte do próprio documento.

O maior tell da edição de campos é a fonte: o texto redesenhado com uma fonte do
sistema nunca bate com a fonte oficial impressa, e o DinoV2 aprende esse descasamento
em vez da fraude. Este módulo evita o problema recompondo o valor novo a partir dos
GLIFOS REAIS já presentes no documento: um banco de recortes por-caractere é colhido
por OCR (caixas por-caractere do Tesseract), e cada letra/dígito do texto novo é
carimbada usando a silhueta de tinta de uma ocorrência real daquele caractere.

É um segundo backend de `edicao_campos` (gerador `transplante_glifo`), ao lado do
render clássico — a diversidade obriga o detector a não decorar nenhuma ferramenta.
Quando a cobertura de caracteres do documento é insuficiente, retorna None para o
chamador cair no render clássico.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract

from scripts.comum import deteccao  # efeito colateral: configura o Tesseract + _IDIOMA_OCR


def _mascara_tinta(crop_bgr):
    """Silhueta de tinta (uint8 0/255) de um recorte de um caractere, ciente de
    polaridade (texto escuro sobre claro ou claro sobre escuro)."""
    cinza = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    p15, med, p85 = np.percentile(cinza, [15, 50, 85])
    texto_escuro = (med - p15) >= (p85 - med)
    if texto_escuro:
        _, m = cv2.threshold(cinza, int((med + p15) / 2), 255, cv2.THRESH_BINARY_INV)
    else:
        _, m = cv2.threshold(cinza, int((med + p85) / 2), 255, cv2.THRESH_BINARY)
    return m


def construir_banco_glifos(imagem_bgr, lang=None) -> dict:
    """Colhe um banco {caractere: [máscara_de_tinta, ...]} dos glifos reais do
    documento, via caixas por-caractere do Tesseract (`image_to_boxes`). Filtra
    fusões de caracteres (caixas largas demais) e outliers de altura (mesclas de
    linhas), que produziriam glifos-lixo. Retorna {} se o OCR falhar. É construído
    UMA vez por documento (custa uma passada de OCR)."""
    lang = lang or deteccao._IDIOMA_OCR
    alt, larg = imagem_bgr.shape[:2]
    try:
        caixas = pytesseract.image_to_boxes(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY), lang=lang)
    except Exception:
        return {}
    brutos = []  # (caractere, máscara, altura_da_caixa)
    for linha in caixas.splitlines():
        partes = linha.split(" ")
        if len(partes) < 5:
            continue
        ch = partes[0]
        if not ch or ch.isspace():
            continue
        try:
            x1, y1, x2, y2 = (int(partes[i]) for i in range(1, 5))
        except ValueError:
            continue
        # image_to_boxes usa origem no canto INFERIOR-esquerdo; converte para topo.
        yt, yb = max(0, alt - y2), min(alt, alt - y1)
        x1c, x2c = max(0, x1), min(larg, x2)
        wb, hb = x2c - x1c, yb - yt
        if wb < 3 or hb < 6:                   # ruído / caractere minúsculo
            continue
        if wb > 1.6 * hb:                      # caixa larga demais → fusão de vários caracteres
            continue
        crop = imagem_bgr[yt:yb, x1c:x2c]
        if crop.size == 0:
            continue
        m = _mascara_tinta(crop)
        if int(m.sum()) < 3 * 255:             # quase sem tinta → provável ruído
            continue
        brutos.append((ch, m, hb))
    if not brutos:
        return {}
    # Descarta outliers de altura (ex.: caixa que mesclou duas linhas) usando a mediana.
    med = float(np.median([h for _, _, h in brutos]))
    banco: dict[str, list] = {}
    for ch, m, hb in brutos:
        if 0.55 * med <= hb <= 1.8 * med:
            banco.setdefault(ch, []).append(m)
    return banco


def _buscar(banco, ch):
    """Procura o caractere no banco tolerando maiúscula/minúscula."""
    for k in (ch, ch.upper(), ch.lower()):
        if banco.get(k):
            return banco[k]
    return None


def cobertura(banco, texto) -> float:
    """Fração dos caracteres não-espaço de `texto` que existem no banco (0..1)."""
    nao_espaco = [c for c in texto if not c.isspace()]
    if not nao_espaco:
        return 0.0
    presentes = sum(1 for c in nao_espaco if _buscar(banco, c) is not None)
    return presentes / len(nao_espaco)


def render_glifos(imagem_bgr, texto, x, y, w, h, cor_bgr, rng, banco, cobertura_min=0.7,
                  harmonizar=True):
    """Recompõe `texto` na região (x,y,w,h) carimbando glifos reais do `banco`, na
    cor `cor_bgr`, em TAMANHO NATURAL (altura ≈ altura do campo) e ALINHADO À ESQUERDA
    como o texto original — um valor mais curto que o original NÃO é esticado para
    preencher o campo (só encolhe se estourar a largura). Retorna a imagem modificada,
    ou None se a cobertura de caracteres for insuficiente. A região deve já ter o texto
    original removido."""
    if not banco or cobertura(banco, texto) < cobertura_min:
        return None
    gh = max(6, int(h * 0.92))                 # glifos ~na altura do campo
    gap = max(1, int(gh * 0.12))
    largura_espaco = max(2, int(gh * 0.5))

    pecas = []  # ("espaco", px) | ("glifo", mascara_escalada)
    for ch in texto:
        if ch.isspace():
            pecas.append(("espaco", largura_espaco)); continue
        opcoes = _buscar(banco, ch)
        if not opcoes:
            pecas.append(("espaco", largura_espaco)); continue   # lacuna sob o limiar
        m = opcoes[rng.randrange(len(opcoes))]
        mh, mw = m.shape[:2]
        if mh == 0 or mw == 0:
            pecas.append(("espaco", largura_espaco)); continue
        nw = max(1, int(mw * gh / mh))
        pecas.append(("glifo", cv2.resize(m, (nw, gh), interpolation=cv2.INTER_AREA)))

    largura = sum((v if t == "espaco" else v.shape[1]) + gap for t, v in pecas)
    largura = max(1, largura - gap)

    alpha = np.zeros((gh, largura), np.float32)
    cx = 0
    for tipo, val in pecas:
        if tipo == "espaco":
            cx += val + gap; continue
        gwv = val.shape[1]
        alpha[:, cx:cx + gwv] = np.maximum(alpha[:, cx:cx + gwv], val.astype(np.float32) / 255.0)
        cx += gwv + gap

    # Encolhe SÓ se estourar a largura do campo (nome curto não é esticado).
    escala = min(1.0, (w * 0.98) / largura)
    nw2, nh2 = max(1, int(largura * escala)), max(1, int(gh * escala))
    alpha_s = cv2.resize(alpha, (nw2, nh2), interpolation=cv2.INTER_AREA)

    resultado = imagem_bgr.copy()
    alt, larg = resultado.shape[:2]
    px = x + max(1, int(h * 0.08))             # alinhado à esquerda, como o original
    py = y + max(0, (h - nh2) // 2)
    nh2, nw2 = min(nh2, alt - py), min(nw2, larg - px)
    if nh2 <= 0 or nw2 <= 0:
        return None
    a = np.clip(alpha_s[:nh2, :nw2], 0, 1)[..., None]
    regiao = resultado[py:py + nh2, px:px + nw2].astype(np.float32)
    cor = np.array(cor_bgr, np.float32)
    resultado[py:py + nh2, px:px + nw2] = np.clip(regiao * (1 - a) + cor * a, 0, 255).astype(np.uint8)

    if harmonizar:
        # Repõe o grão de fundo que falta na região reescrita (anti-atalho), igual
        # ao caminho do desenhar_texto.
        from scripts.comum.textura import harmonizar_textura
        margem = max(2, h // 6)
        resultado = harmonizar_textura(resultado, x - margem, y - margem,
                                       w + 2 * margem, h + 2 * margem, rng)
    return resultado
