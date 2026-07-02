"""Técnica 6 — Reimpressão (print/recapture attack).

LIMITAÇÃO IMPORTANTE (mesma nota de `recaptura_tela.py`): o artefato real de
reimpressão — textura de halftone/dithering da impressora, absorção de tinta
no papel, curvatura leve da folha — só existe de fato ao imprimir o
documento (ou uma versão editada dele) em papel comum e refotografá-lo,
replicando a metodologia do dataset acadêmico FantasyID. Este script produz
uma APROXIMAÇÃO SINTÉTICA computacional, útil como aumento de dados
complementar, não substituto do processo físico real.

Efeitos simulados:
    - dithering ordenado (padrão Bayer 4x4), aproximando o halftone de
      impressão;
    - leve dessaturação (tinta não reproduz a gama de cores da tela);
    - suavização de borda (leve desfoque, simulando espalhamento de tinta no
      papel);
    - textura de papel (ruído correlacionado de baixa frequência) e
      iluminação levemente desigual (gradiente radial suave, simulando
      fotografar uma folha física sob luz ambiente).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

_MATRIZ_BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32) / 16.0


def _dithering_ordenado(canal_cinza_normalizado: np.ndarray) -> np.ndarray:
    altura, largura = canal_cinza_normalizado.shape
    limiar = np.tile(_MATRIZ_BAYER_4X4, (altura // 4 + 1, largura // 4 + 1))[:altura, :largura]
    return (canal_cinza_normalizado > limiar).astype(np.float32)


def _textura_papel(altura: int, largura: int, rng: random.Random) -> np.ndarray:
    # `rng` é um random.Random (não numpy) para manter a mesma semente
    # determinística usada no resto do módulo; deriva um gerador numpy a
    # partir dele para gerar o campo de ruído 2D.
    gerador_numpy = np.random.default_rng(rng.randint(0, 2**31 - 1))
    ruido = gerador_numpy.random(size=(altura // 8 + 1, largura // 8 + 1)).astype(np.float32)
    ruido_suave = cv2.resize(ruido, (largura, altura), interpolation=cv2.INTER_CUBIC)
    return (ruido_suave - ruido_suave.mean())


def aplicar(imagem_bgr: np.ndarray, rng: random.Random, intensidade_dithering: float = 0.25) -> tuple[np.ndarray, dict]:
    altura, largura = imagem_bgr.shape[:2]
    imagem_float = imagem_bgr.astype(np.float32)

    # 1. Dessaturação leve (tinta CMYK não cobre a gama RGB da tela)
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    fator_dessaturacao = rng.uniform(0.75, 0.9)
    hsv[..., 1] *= fator_dessaturacao
    imagem_dessaturada = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 2. Halftone sutil: mistura a imagem original com uma versão "ditherizada"
    #    em baixa intensidade — um halftone 100% substituiria a imagem, o que
    #    seria irreal para uma foto (não um scan) de um documento impresso.
    cinza_normalizado = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dither = _dithering_ordenado(cinza_normalizado) * 255.0
    dither_3c = np.repeat(dither[..., None], 3, axis=2)
    imagem_com_halftone = (
        imagem_dessaturada * (1 - intensidade_dithering) + dither_3c * intensidade_dithering
    )

    # 3. Textura de papel: ruído correlacionado de baixa frequência
    textura = _textura_papel(altura, largura, rng)[..., None] * rng.uniform(4, 9)
    imagem_com_textura = imagem_com_halftone + textura

    # 4. Iluminação desigual: gradiente radial suave centrado num ponto aleatório
    cx, cy = rng.uniform(0.3, 0.7) * largura, rng.uniform(0.3, 0.7) * altura
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    distancia = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    distancia_normalizada = distancia / distancia.max()
    sombreamento = 1.0 - distancia_normalizada * rng.uniform(0.08, 0.18)
    imagem_com_sombra = imagem_com_textura * sombreamento[..., None]

    # 5. Leve desfoque (espalhamento de tinta no papel)
    resultado = np.clip(imagem_com_sombra, 0, 255).astype(np.uint8)
    resultado = cv2.GaussianBlur(resultado, (3, 3), sigmaX=0.5)

    parametros = {
        "fator_dessaturacao": float(fator_dessaturacao),
        "intensidade_dithering": intensidade_dithering,
        "aviso": "aproximacao_sintetica_nao_substitui_reimpressao_real",
    }
    return resultado, parametros


def processar_documento(caminho: Path, tipo_documento: str, pasta_saida: Path, rng: random.Random, variantes: int) -> int:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0

    pasta_saida_tecnica = pasta_saida / "reimpressao"
    pasta_saida_tecnica.mkdir(parents=True, exist_ok=True)

    total = 0
    for i in range(variantes):
        intensidade_dithering = rng.uniform(0.15, 0.35)
        resultado, parametros = aplicar(imagem, rng, intensidade_dithering)
        nome_arquivo = f"{caminho.stem}__reimpressao_{i}.jpg"
        cv2.imwrite(str(pasta_saida_tecnica / nome_arquivo), resultado, [cv2.IMWRITE_JPEG_QUALITY, 85])

        registro = RegistroFraude(
            tecnica="reimpressao",
            documento_origem=str(caminho),
            tipo_documento=tipo_documento,
            arquivo_gerado=nome_arquivo,
            parametros=parametros,
            campo_alterado=None,
            dificuldade="sutil",
        )
        registro.salvar(pasta_saida_tecnica)
        total += 1
    return total


def processar_dataset(pasta_legitimos: Path, pasta_saida: Path, variantes_por_documento: int, semente: int = 42) -> int:
    rng = random.Random(semente)
    total = 0
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        for caminho in sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM):
            total += processar_documento(caminho, pasta_tipo.name, pasta_saida, rng, variantes_por_documento)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/fraude_gerada")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"reimpressao: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'reimpressao'}")


if __name__ == "__main__":
    main()
