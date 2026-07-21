"""Ruído de sensor de câmera fisicamente calibrado, usado pelo simulador de
captura compartilhado (`scripts/comum/captura.py`).

O ruído de um sensor real é heterocedástico: a variância cresce com o sinal
(shot noise) mais um componente fixo de leitura (read noise) — não um gaussiano
de variância constante. Este é o modelo padrão em fotografia computacional
(Foi et al.).
"""

from __future__ import annotations

import random

import numpy as np


def ruido_sensor_poisson_gaussiano(
    imagem_float_0_255: np.ndarray, ganho_iso: float, rng: random.Random
) -> np.ndarray:
    """Ruído heterocedástico de sensor: variância = a*sinal + b (shot + read),
    com `ganho_iso` controlando a intensidade geral (análogo a ISO mais alto —
    luz ambiente comum de foto de celular, não estúdio controlado)."""
    gerador = np.random.default_rng(rng.randint(0, 2**31 - 1))
    sinal = np.clip(imagem_float_0_255, 0, 255)
    variancia_shot = sinal * (ganho_iso * 0.15)
    variancia_read = (ganho_iso * 1.2) ** 2
    desvio_total = np.sqrt(variancia_shot + variancia_read)
    ruido = gerador.standard_normal(size=imagem_float_0_255.shape).astype(np.float32) * desvio_total
    return imagem_float_0_255 + ruido
