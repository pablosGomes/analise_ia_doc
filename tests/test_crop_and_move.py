import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.geracao_fraude.crop_and_move import aplicar, _escolher_regiao, _sobrepoe


def _imagem_teste(altura=300, largura=400):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(altura, largura, 3), dtype=np.uint8)


def test_sobrepoe_detecta_intersecao():
    assert _sobrepoe((0, 0, 10, 10), (5, 5, 10, 10)) is True
    assert _sobrepoe((0, 0, 10, 10), (20, 20, 10, 10)) is False


def test_escolher_regiao_respeita_limites_da_imagem():
    rng = random.Random(0)
    regiao = _escolher_regiao(400, 300, rng, evitar=[])
    assert regiao is not None
    x, y, w, h = regiao
    assert 0 <= x and x + w <= 400
    assert 0 <= y and y + h <= 300


def test_escolher_regiao_evita_area_indicada():
    rng = random.Random(0)
    area_proibida = (0, 0, 400, 300)  # ocupa a imagem inteira
    regiao = _escolher_regiao(400, 300, rng, evitar=[area_proibida], tentativas=5)
    assert regiao is None  # não deve existir espaço livre


def test_aplicar_duplicacao_preserva_dimensoes():
    imagem = _imagem_teste()
    rng = random.Random(0)
    resultado = aplicar(imagem, "duplicacao", rng, evitar=[])
    assert resultado is not None
    imagem_resultado, parametros = resultado
    assert imagem_resultado.shape == imagem.shape
    assert parametros["modo"] == "duplicacao"


def test_aplicar_deslocamento_altera_regiao_origem():
    imagem = _imagem_teste()
    rng = random.Random(3)
    resultado = aplicar(imagem, "deslocamento", rng, evitar=[])
    assert resultado is not None
    imagem_resultado, parametros = resultado
    ox, oy, ow, oh = parametros["origem"]
    # A região de origem deve ter sido preenchida (não é mais idêntica ao original)
    original_recorte = imagem[oy:oy + oh, ox:ox + ow]
    novo_recorte = imagem_resultado[oy:oy + oh, ox:ox + ow]
    assert not np.array_equal(original_recorte, novo_recorte)
