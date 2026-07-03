"""Modelos de ruído/artefato fisicamente calibrados, compartilhados entre
`recaptura_tela.py` e `reimpressao.py` — substituem aproximações mais
ingênuas (blur gaussiano genérico, senoide simples) por modelos que refletem
mais de perto o mecanismo físico real de cada artefato:

    - `ruido_sensor_poisson_gaussiano`: ruído de sensor de câmera real é
      heterocedástico (variância cresce com o sinal — shot noise — mais um
      componente fixo de ruído de leitura), não um gaussiano de variância
      constante. Modelo padrão em fotografia computacional (Foi et al.).
    - `moire_por_subamostragem`: moiré real nasce do batimento entre duas
      grades periódicas quase alinhadas (a grade de subpixels da tela e a
      grade de amostragem do sensor da câmera) — reproduzido aqui via uma
      reamostragem down→up genuína (que produz aliasing real), em vez de
      somar uma senoide arbitrária diretamente sobre a imagem.
    - `cascata_recompressao_jpeg`: uma foto de recaptura/reimpressão passa
      por várias gerações de compressão JPEG (documento original → exibição
      → nova foto → upload); cada passada acumula blocagem 8x8 característica,
      artefato que um único blur gaussiano não reproduz.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def ruido_sensor_poisson_gaussiano(
    imagem_float_0_255: np.ndarray, ganho_iso: float, rng: random.Random
) -> np.ndarray:
    """Ruído heterocedástico de sensor: variância = a*sinal + b (shot noise +
    read noise), com `ganho_iso` controlando a intensidade geral (análogo a
    fotografar em ISO mais alto — luz ambiente comum ao refotografar uma
    tela ou uma folha impressa, não um estúdio controlado)."""
    gerador = np.random.default_rng(rng.randint(0, 2**31 - 1))
    sinal = np.clip(imagem_float_0_255, 0, 255)
    variancia_shot = sinal * (ganho_iso * 0.15)
    variancia_read = (ganho_iso * 1.2) ** 2
    desvio_total = np.sqrt(variancia_shot + variancia_read)
    ruido = gerador.standard_normal(size=imagem_float_0_255.shape).astype(np.float32) * desvio_total
    return imagem_float_0_255 + ruido


def moire_por_subamostragem(
    altura: int, largura: int, rng: random.Random,
    freq_grade_px: float | None = None, fator_reamostragem: float | None = None,
) -> np.ndarray:
    """Gera um padrão de moiré via reamostragem genuína (down-scale seguido
    de up-scale) de uma grade periódica fina — o aliasing que sobra dessa
    reamostragem É o mecanismo físico real do moiré (batimento entre a grade
    de subpixels da tela e a grade de amostragem da câmera), diferente de
    uma senoide desenhada diretamente na frequência "certa" a olho.
    Retorna um campo centrado em zero (mesma faixa de uso de antes)."""
    fase = rng.uniform(0, 2 * np.pi)
    freq_grade_px = freq_grade_px if freq_grade_px is not None else rng.uniform(2.5, 3.8)
    angulo = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    projecao = x * np.cos(angulo) + y * np.sin(angulo)
    grade_fina = 0.5 + 0.5 * np.sin(2 * np.pi * projecao / freq_grade_px + fase)

    fator = fator_reamostragem if fator_reamostragem is not None else rng.uniform(0.35, 0.65)
    largura_reduzida = max(1, int(largura * fator))
    altura_reduzida = max(1, int(altura * fator))
    grade_reduzida = cv2.resize(grade_fina, (largura_reduzida, altura_reduzida), interpolation=cv2.INTER_LINEAR)
    padrao_moire = cv2.resize(grade_reduzida, (largura, altura), interpolation=cv2.INTER_LINEAR)

    return padrao_moire - float(padrao_moire.mean())


def cascata_recompressao_jpeg(
    imagem_uint8: np.ndarray, qualidades: list[int]
) -> np.ndarray:
    """Recomprime a imagem sucessivamente nas qualidades JPEG informadas,
    simulando a cadeia real de gerações de compressão de uma foto de
    recaptura/reimpressão (documento → exibição/impressão → nova foto →
    upload). Cada passada reintroduz blocagem 8x8, acumulando o artefato de
    forma mais realista que um único blur."""
    resultado = imagem_uint8
    for qualidade in qualidades:
        ok, buffer = cv2.imencode(".jpg", resultado, [cv2.IMWRITE_JPEG_QUALITY, int(qualidade)])
        if not ok:
            continue
        decodificado = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decodificado is not None:
            resultado = decodificado
    return resultado
