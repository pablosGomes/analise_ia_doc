"""Detecção de rosto e malha de landmarks faciais via MediaPipe (API Tasks).

Substitui a detecção por Haar Cascade para a técnica de troca de foto: no OpenCV 5
o Haar foi separado em pacotes incompatíveis e ficou indisponível nesta máquina, e
o MediaPipe é mais robusto de qualquer forma (alinha olhos/nariz/boca, não só uma
caixa). A foto de rosto num documento é pequena em relação à página inteira, então
a detecção testa as quatro rotações de 90° (fotos de celular raramente estão "em
pé") e escolhe a de maior confiança; os landmarks são extraídos de um recorte do
rosto, onde ele é grande o bastante para a malha.

Modelos (baixados uma vez para `models/`, gitignored):
    - blaze_face_short_range.tflite  (detector de rosto)
    - face_landmarker.task           (malha de 478 pontos)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_MODELOS = Path(__file__).resolve().parents[2] / "models"

# Modelos do MediaPipe (baixados uma vez para models/, que é gitignored). O
# download é automático no primeiro uso, então um clone novo funciona sem passo
# manual.
_URLS_MODELOS = {
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def _caminho_modelo(nome: str) -> str:
    caminho = _MODELOS / nome
    if not caminho.exists():
        import urllib.request
        _MODELOS.mkdir(parents=True, exist_ok=True)
        print(f"[rosto] baixando modelo {nome}...")
        # Baixa para um arquivo temporário e só então renomeia: um download
        # interrompido não deixa um .task/.tflite parcial em cache, que travaria
        # silenciosamente todas as execuções seguintes do MediaPipe.
        tmp = caminho.with_name(caminho.name + ".tmp")
        try:
            urllib.request.urlretrieve(_URLS_MODELOS[nome], tmp)
            tmp.replace(caminho)
        finally:
            if tmp.exists():
                tmp.unlink()
    return str(caminho)


_ROTACOES = [
    (0, None, None),
    (90, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),
    (180, cv2.ROTATE_180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE),
]

_detector = None
_landmarker = None


def _obter_detector():
    global _detector
    if _detector is None:
        _detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_caminho_modelo("blaze_face_short_range.tflite")),
                min_detection_confidence=0.3))
    return _detector


def _obter_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_caminho_modelo("face_landmarker.task")),
                num_faces=1))
    return _landmarker


def _mp_image(imagem_bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB))


@dataclass
class Rosto:
    """Rosto localizado num documento, já com a imagem rotacionada para ficar em pé."""
    imagem: np.ndarray            # documento na rotação em que o rosto ficou de pé
    codigo_rotacao_inversa: int   # cv2.ROTATE_* para desfazer a rotação (ou None)
    caixa: tuple[int, int, int, int]   # (x, y, w, h) do rosto em `imagem`
    landmarks: np.ndarray         # (478, 2) em coordenadas de `imagem`
    confianca: float


def _detectar_melhor_rotacao(imagem_bgr):
    """Testa as 4 rotações e retorna a de maior confiança: (imagem_rot, inversa, caixa, score)."""
    detector = _obter_detector()
    melhor = None
    for _ang, codigo, inversa in _ROTACOES:
        rot = imagem_bgr if codigo is None else cv2.rotate(imagem_bgr, codigo)
        deteccoes = detector.detect(_mp_image(rot)).detections
        if not deteccoes:
            continue
        d = deteccoes[0]
        score = d.categories[0].score
        if melhor is None or score > melhor[3]:
            bb = d.bounding_box
            melhor = (rot, inversa, (bb.origin_x, bb.origin_y, bb.width, bb.height), score)
    return melhor


def localizar(imagem_bgr) -> Rosto | None:
    """Localiza o rosto no documento e extrai a malha de 478 landmarks. Retorna None
    se não achar rosto ou não conseguir a malha."""
    melhor = _detectar_melhor_rotacao(imagem_bgr)
    if melhor is None:
        return None
    rot, inversa, (x, y, w, h), score = melhor

    # Recorta o rosto com margem para a malha (o landmarker precisa do rosto grande).
    margem = int(max(w, h) * 0.5)
    alt, larg = rot.shape[:2]
    rx0, ry0 = max(0, x - margem), max(0, y - margem)
    rx1, ry1 = min(larg, x + w + margem), min(alt, y + h + margem)
    recorte = rot[ry0:ry1, rx0:rx1]
    if recorte.size == 0:
        return None

    resultado = _obter_landmarker().detect(_mp_image(recorte))
    if not resultado.face_landmarks:
        return None

    rh, rw = recorte.shape[:2]
    pontos = np.array([[lm.x * rw + rx0, lm.y * rh + ry0]
                       for lm in resultado.face_landmarks[0]], dtype=np.float32)
    return Rosto(imagem=rot, codigo_rotacao_inversa=inversa,
                 caixa=(x, y, w, h), landmarks=pontos, confianca=float(score))


# Índices de landmarks estáveis da malha canônica do MediaPipe, usados como
# referência do alinhamento em troca_foto._PONTOS_ALINHAMENTO.
OLHO_ESQ, OLHO_DIR = 33, 263      # cantos externos dos olhos
NARIZ = 1                          # ponta do nariz
BOCA_ESQ, BOCA_DIR = 61, 291      # cantos da boca
