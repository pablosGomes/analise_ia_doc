"""Detecção de campos de texto em documentos de identidade (via OCR).

Usado pela edição de campos para localizar o valor associado a um rótulo (ex.:
"NOME", "FILIAÇÃO"). A detecção testa as quatro rotações da imagem — fotos de
celular frequentemente não estão "em pé" — e une os achados de volta no
referencial original por máscara rotacionada, nunca por contas manuais de
coordenadas (que são frágeis).

A detecção de ROSTO (usada pela troca de foto) fica em `scripts/comum/rosto.py`,
via MediaPipe; não faz parte deste módulo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def _configurar_tesseract_windows() -> None:
    """No Windows o binário do Tesseract normalmente não fica no PATH após a
    instalação (ex.: via winget/UB-Mannheim). Se não estiver no PATH, aponta o
    pytesseract para os caminhos de instalação padrão. No Linux (tesseract no
    PATH) é no-op, preservando a paridade de comportamento entre os ambientes."""
    import shutil

    if os.name != "nt" or shutil.which("tesseract"):
        return
    candidatos = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Tesseract-OCR\tesseract.exe"),
    ]
    caminho_env = os.environ.get("TESSERACT_CMD")
    if caminho_env:
        candidatos.insert(0, caminho_env)
    for cand in candidatos:
        if cand and Path(cand).exists():
            pytesseract.pytesseract.tesseract_cmd = cand
            return


_configurar_tesseract_windows()


def _idioma_ocr() -> str:
    """Prefere português (documentos BR têm acentos: FILIAÇÃO, NAÇÃO) quando o
    pacote estiver instalado; cai para inglês caso contrário. Para ativar, instale
    `por.traineddata` na pasta tessdata do Tesseract."""
    try:
        return "por" if "por" in pytesseract.get_languages(config="") else "eng"
    except Exception:
        return "eng"


_IDIOMA_OCR = _idioma_ocr()

ROTACOES = [
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
]
_INVERSA = {
    None: None,
    cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180: cv2.ROTATE_180,
}


@dataclass(frozen=True)
class CaixaDelimitadora:
    x: int
    y: int
    w: int
    h: int
    motivo: str = ""
    texto: str = ""

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def centro(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def como_tupla(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _rotacionar(img: np.ndarray, codigo) -> np.ndarray:
    return img if codigo is None else cv2.rotate(img, codigo)


def _uniao_multi_rotacao(img_original: np.ndarray, obter_regioes) -> list[CaixaDelimitadora]:
    """Roda `obter_regioes(img_rotacionada) -> list[CaixaDelimitadora]` nas 4
    rotações e une os achados no referencial da imagem ORIGINAL, via máscara
    binária rotacionada de volta (não por conta manual de coordenadas).

    Para preservar `texto`/`motivo` de cada região (necessário porque a detecção
    depende do texto OCR'ado), cada região recebe um ID inteiro único, desenhado
    como um "mapa de rótulos" rotacionado de volta pela mesma operação exata
    (`cv2.rotate`, sem interpolação — lossless para múltiplos de 90°) usada na
    máscara. Isso evita deduzir manualmente a transformação de coordenadas por
    rotação, uma fonte comum de bugs sutis.
    """
    alt, larg = img_original.shape[:2]
    mascara_total = np.zeros((alt, larg), dtype=np.uint8)
    rotulo_final = np.zeros((alt, larg), dtype=np.int32)
    info_por_id: dict[int, CaixaDelimitadora] = {}
    proximo_id = 1

    for _angulo, codigo in ROTACOES:
        img_rot = _rotacionar(img_original, codigo)
        regioes = obter_regioes(img_rot)
        if not regioes:
            continue

        mascara_rot = np.zeros(img_rot.shape[:2], dtype=np.uint8)
        rotulo_rot = np.zeros(img_rot.shape[:2], dtype=np.int32)
        for r in regioes:
            rid = proximo_id
            proximo_id += 1
            info_por_id[rid] = r
            x0, y0 = max(0, r.x - 3), max(0, r.y - 3)
            x1, y1 = r.x + r.w + 3, r.y + r.h + 3
            cv2.rectangle(mascara_rot, (x0, y0), (x1, y1), 255, -1)
            cv2.rectangle(rotulo_rot, (x0, y0), (x1, y1), rid, -1)

        mascara_de_volta = _rotacionar(mascara_rot, _INVERSA[codigo])
        rotulo_de_volta = _rotacionar(rotulo_rot, _INVERSA[codigo])
        mascara_total = cv2.bitwise_or(mascara_total, mascara_de_volta)

        # Só preenche onde ainda não há rótulo — dá prioridade às rotações
        # processadas antes (a rotação 0 é a primeira de ROTACOES, então é
        # preferida quando o mesmo achado aparece em mais de uma rotação).
        preencher = (rotulo_final == 0) & (rotulo_de_volta != 0)
        rotulo_final[preencher] = rotulo_de_volta[preencher]

    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        sub_rotulo = rotulo_final[y:y + h, x:x + w]
        valores = sub_rotulo[sub_rotulo != 0]
        if valores.size > 0:
            ids_unicos, contagens = np.unique(valores, return_counts=True)
            melhor_id = int(ids_unicos[np.argmax(contagens)])
            origem = info_por_id[melhor_id]
            resultado.append(CaixaDelimitadora(x=x, y=y, w=w, h=h, motivo=origem.motivo, texto=origem.texto))
        else:
            resultado.append(CaixaDelimitadora(x=x, y=y, w=w, h=h))
    return resultado


def _dados_ocr(img_bgr: np.ndarray) -> dict:
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_data(cinza, lang=_IDIOMA_OCR, output_type=Output.DICT)


def detectar_valores_proximos_a_rotulo(imagem_bgr: np.ndarray, palavras_chave: list[str]) -> list[CaixaDelimitadora]:
    """Detecta o(s) valor(es) associado(s) a um rótulo textual (ex.: "NOME",
    "NASCIMENTO", "VALIDADE"). Cobre dois layouts comuns em documentos
    brasileiros: rótulo e valor na mesma linha, ou rótulo numa linha e valor(es)
    em até 2 linhas abaixo dentro do mesmo bloco (ex.: FILIAÇÃO com pai e mãe em
    linhas separadas).
    """
    palavras_chave_upper = [p.upper() for p in palavras_chave]

    def _regioes(img: np.ndarray) -> list[CaixaDelimitadora]:
        dados = _dados_ocr(img)
        n = len(dados["text"])

        linhas: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            chave = (dados["block_num"][i], dados["line_num"][i])
            linhas.setdefault(chave, []).append(i)

        linhas_ordenadas: dict[int, list[int]] = {}
        for (bloco, linha) in linhas:
            linhas_ordenadas.setdefault(bloco, []).append(linha)
        for bloco in linhas_ordenadas:
            linhas_ordenadas[bloco] = sorted(set(linhas_ordenadas[bloco]))

        regioes = []
        for (bloco, linha), indices in linhas.items():
            for i in indices:
                texto_upper = dados["text"][i].strip().upper()
                if not any(kw in texto_upper for kw in palavras_chave_upper):
                    continue

                direita_rotulo = dados["left"][i] + dados["width"][i]
                candidatos_mesma_linha = [j for j in indices if dados["left"][j] > direita_rotulo]
                if candidatos_mesma_linha:
                    x_min = min(dados["left"][j] for j in candidatos_mesma_linha)
                    x_max = max(dados["left"][j] + dados["width"][j] for j in candidatos_mesma_linha)
                    y_min = min(dados["top"][j] for j in candidatos_mesma_linha + [i])
                    y_max = max(dados["top"][j] + dados["height"][j] for j in candidatos_mesma_linha + [i])
                    texto_valor = " ".join(dados["text"][j] for j in candidatos_mesma_linha).strip()
                    regioes.append(CaixaDelimitadora(
                        x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min,
                        motivo=f"valor_apos_rotulo:{texto_upper}", texto=texto_valor,
                    ))

                linhas_do_bloco = linhas_ordenadas.get(bloco, [])
                if linha in linhas_do_bloco:
                    pos = linhas_do_bloco.index(linha)
                    for prox_linha in linhas_do_bloco[pos + 1: pos + 3]:
                        indices_prox = linhas.get((bloco, prox_linha), [])
                        textos_prox = [dados["text"][j].strip() for j in indices_prox if dados["text"][j].strip()]
                        if not textos_prox:
                            continue
                        if any(kw in t.upper() for t in textos_prox for kw in palavras_chave_upper):
                            break
                        x_min = min(dados["left"][j] for j in indices_prox)
                        x_max = max(dados["left"][j] + dados["width"][j] for j in indices_prox)
                        y_min = min(dados["top"][j] for j in indices_prox)
                        y_max = max(dados["top"][j] + dados["height"][j] for j in indices_prox)
                        regioes.append(CaixaDelimitadora(
                            x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min,
                            motivo=f"linha_abaixo_de:{texto_upper}", texto=" ".join(textos_prox),
                        ))
        return regioes

    return _uniao_multi_rotacao(imagem_bgr, _regioes)


def maior_caixa(caixas: list[CaixaDelimitadora]) -> CaixaDelimitadora | None:
    """Retorna a caixa de maior área — heurística útil quando se espera um único
    alvo mas o detector pode achar falsos positivos menores."""
    if not caixas:
        return None
    return max(caixas, key=lambda c: c.w * c.h)
