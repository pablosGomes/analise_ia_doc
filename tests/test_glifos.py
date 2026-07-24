import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.comum import glifos


def _banco_falso(chars):
    """Banco sintético: cada caractere mapeia para uma máscara de tinta retangular."""
    m = np.zeros((20, 12), np.uint8)
    m[3:17, 2:10] = 255
    return {c: [m] for c in chars}


def test_cobertura():
    banco = _banco_falso("ABC123")
    assert glifos.cobertura(banco, "AB1") == 1.0
    assert glifos.cobertura(banco, "ABX") < 1.0        # X ausente
    assert glifos.cobertura(banco, "abc") == 1.0       # tolera maiúscula/minúscula
    assert glifos.cobertura({}, "AB") == 0.0


def test_render_glifos_compoe_quando_ha_cobertura():
    banco = _banco_falso("ABC 123")
    img = np.full((100, 300, 3), 240, np.uint8)        # fundo claro
    res = glifos.render_glifos(img, "ABC", 50, 40, 120, 24, (20, 20, 20),
                               random.Random(0), banco, harmonizar=False)
    assert res is not None
    assert res.shape == img.shape
    # a região editada deve ter recebido tinta (pixels escurecidos)
    assert res[40:64, 50:170].min() < 200


def test_render_glifos_none_sem_cobertura():
    banco = _banco_falso("XYZ")
    img = np.full((100, 300, 3), 240, np.uint8)
    res = glifos.render_glifos(img, "ABC", 50, 40, 120, 24, (20, 20, 20),
                               random.Random(0), banco)
    assert res is None                                  # cobertura 0 < limiar


def test_render_glifos_none_banco_vazio():
    img = np.full((100, 300, 3), 240, np.uint8)
    assert glifos.render_glifos(img, "ABC", 50, 40, 120, 24, (20, 20, 20),
                                random.Random(0), {}) is None
