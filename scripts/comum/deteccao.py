"""Detecção de regiões de interesse em documentos de identidade — módulo
compartilhado entre `geracao_fraude/`, `anonimizacao/` e `validacao_dados/`.

Adaptado do script `anonimizar_documento.py` do projeto `prototipo_Ia_doc`
(mesma estratégia comprovada: testar as 4 rotações da imagem, já que fotos
reais de celular frequentemente não estão "em pé", e unir os achados de cada
rotação de volta no referencial original via máscara rotacionada — nunca por
contas manuais de coordenadas, que é frágil).

Fornece três detectores:
    - `detectar_rostos`: bounding boxes de rosto (MediaPipe se disponível,
      sempre + Haar Cascade, união das duas fontes — recall alto é mais
      importante que precisão aqui).
    - `detectar_numeros_documento`: bounding boxes de sequências de 7+
      dígitos (CPF, RG, número de registro, etc.), com o texto OCR'ado.
    - `detectar_valores_proximos_a_rotulo`: bounding boxes do valor associado
      a um rótulo textual (ex.: "NOME", "NASCIMENTO"), cobrindo tanto o
      layout "rótulo: valor na mesma linha" quanto "rótulo em cima, valor(es)
      embaixo".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

REGEX_NUMERO_DOCUMENTO = re.compile(r"\d{7,}")

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


def _limpar_digitos(texto: str) -> str:
    return re.sub(r"\D", "", texto)


def _carregar_face_detector():
    """Tenta carregar o MediaPipe BlazeFace; retorna None se indisponível
    (biblioteca não instalada, libs nativas do sistema faltando, ou modelo
    .tflite não encontrado). Haar Cascade sempre roda como camada extra,
    independente deste resultado.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        caminho_modelo = os.environ.get(
            "MEDIAPIPE_FACE_MODEL",
            str(Path(__file__).parent.parent / "modelos" / "blaze_face_short_range.tflite"),
        )
        if not Path(caminho_modelo).exists():
            return None
        base_options = mp_python.BaseOptions(model_asset_path=caminho_modelo)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.15)
        detector = vision.FaceDetector.create_from_options(options)

        def _detectar(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = detector.detect(mp_image)
            return [
                (d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
                for d in resultado.detections
            ]

        return _detectar
    except Exception:
        return None


_DETECTOR_MEDIAPIPE = _carregar_face_detector()


def usando_mediapipe() -> bool:
    return _DETECTOR_MEDIAPIPE is not None


def _detectar_rostos_uma_rotacao(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    caixas: list[tuple[int, int, int, int]] = []

    if _DETECTOR_MEDIAPIPE is not None:
        try:
            caixas.extend(_DETECTOR_MEDIAPIPE(img_bgr))
        except Exception:
            pass

    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Roda tanto na imagem em tons de cinza crua quanto na equalizada, e une
    # os resultados. Descoberto empiricamente (2026-07-02, dataset de teste
    # sintético): equalização de histograma AJUDA em fotos com iluminação
    # ruim, mas pode PIORAR o contraste relativo da foto de rosto (pequena)
    # quando o documento tem grandes áreas de fundo claro/uniforme (comum em
    # RG/CNH) — nesse caso, detectar na imagem crua funciona melhor. Rodar
    # as duas é mais caro, mas prioriza recall, consistente com o resto do
    # módulo.
    for variante in (cinza, cv2.equalizeHist(cinza)):
        for (x, y, w, h) in cascade.detectMultiScale(variante, 1.08, 4, minSize=(30, 30)):
            caixas.append((int(x), int(y), int(w), int(h)))

    return caixas


def _uniao_multi_rotacao(img_original: np.ndarray, obter_regioes) -> list[CaixaDelimitadora]:
    """Roda `obter_regioes(img_rotacionada) -> list[CaixaDelimitadora]` nas 4
    rotações e une os achados no referencial da imagem ORIGINAL, via máscara
    binária rotacionada de volta (não por conta manual de coordenadas).

    Para preservar `texto`/`motivo` de cada região (necessário para
    `detectar_numeros_documento` e `detectar_valores_proximos_a_rotulo`, que
    dependem do texto OCR'ado), cada região recebe um ID inteiro único,
    desenhado como um "mapa de rótulos" que é rotacionado de volta pela
    mesma operação exata (`cv2.rotate`, sem interpolação — lossless para
    múltiplos de 90°) usada na máscara binária. Isso evita ter que deduzir
    manualmente a fórmula de transformação de coordenadas por rotação, que é
    uma fonte comum de bugs sutis.
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

        # Só preenche onde ainda não há rótulo — dá prioridade a rotações
        # processadas antes (a rotação 0/sem distorção é a primeira da lista
        # ROTACOES, então é preferida quando o mesmo achado aparece em mais
        # de uma rotação).
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


def detectar_rostos(imagem_bgr: np.ndarray) -> list[CaixaDelimitadora]:
    """Detecta rostos na imagem, testando as 4 rotações e unindo os achados.
    Prioriza recall (melhor detectar uma área maior do que perder o rosto)."""
    return _uniao_multi_rotacao(
        imagem_bgr,
        lambda img: [
            CaixaDelimitadora(x=x, y=y, w=w, h=h, motivo="rosto")
            for (x, y, w, h) in _detectar_rostos_uma_rotacao(img)
        ],
    )


def _dados_ocr(img_bgr: np.ndarray) -> dict:
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_data(cinza, lang="eng", output_type=Output.DICT)


def detectar_numeros_documento(imagem_bgr: np.ndarray) -> list[CaixaDelimitadora]:
    """Detecta bounding boxes de sequências de 7+ dígitos (CPF, RG, número de
    registro/CNH, datas com formatação removida pelo OCR, etc.)."""

    def _regioes(img: np.ndarray) -> list[CaixaDelimitadora]:
        dados = _dados_ocr(img)
        regioes = []
        for i in range(len(dados["text"])):
            texto = dados["text"][i].strip()
            digitos = _limpar_digitos(texto)
            if len(digitos) >= 7:
                regioes.append(CaixaDelimitadora(
                    x=dados["left"][i], y=dados["top"][i],
                    w=dados["width"][i], h=dados["height"][i],
                    motivo="numero_documento", texto=texto,
                ))
        return regioes

    return _uniao_multi_rotacao(imagem_bgr, _regioes)


def detectar_valores_proximos_a_rotulo(imagem_bgr: np.ndarray, palavras_chave: list[str]) -> list[CaixaDelimitadora]:
    """Detecta o(s) valor(es) associado(s) a um rótulo textual (ex.: "NOME",
    "NASCIMENTO", "VALIDADE"). Cobre dois layouts comuns em documentos
    brasileiros: rótulo e valor na mesma linha, ou rótulo numa linha e
    valor(es) em até 2 linhas abaixo dentro do mesmo bloco (ex.: FILIAÇÃO
    listando pai e mãe em linhas separadas).
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
    """Retorna a caixa de maior área — heurística útil quando se espera um
    único alvo (ex.: o rosto principal do documento) mas o detector pode
    achar falsos positivos menores."""
    if not caixas:
        return None
    return max(caixas, key=lambda c: c.w * c.h)
