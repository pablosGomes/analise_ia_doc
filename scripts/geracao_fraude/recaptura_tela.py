"""Técnica 5 — Recaptura de tela (screen/replay attack).

LIMITAÇÃO IMPORTANTE (documentada em
docs/documentacao_deteccao_fraude.docx, Seção 2.2): o artefato real de
recaptura de tela — moiré, reflexo especular, padrão de subpixel — só existe
de fato quando alguém fotografa um documento genuíno sendo exibido em uma
tela física. Este script produz uma APROXIMAÇÃO SINTÉTICA computacional
desse efeito, útil como aumento de dados e para dar ao classificador um
primeiro sinal do tipo de artefato, mas não substitui capturar exemplos
reais (fotografar a tela do celular/monitor exibindo o documento com outra
câmera) — recomendado como complemento manual futuro.

Efeitos simulados:
    - padrão de moiré: modulação de luminância senoidal de alta frequência,
      aproximando o aliasing da grade de subpixels de uma tela;
    - leve deslocamento de canal de cor (1-2px), simulando aberração
      cromática e desalinhamento de subpixel RGB;
    - brilho/gradiente de reflexo especular parcial;
    - leve desfoque gaussiano (perda de nitidez por refotografar uma
      superfície emissora de luz através do ar).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def _padrao_moire(altura: int, largura: int, rng: random.Random) -> np.ndarray:
    frequencia = rng.uniform(0.35, 0.6)  # ciclos por pixel — alta frequência, sutil
    angulo = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    projecao = x * np.cos(angulo) + y * np.sin(angulo)
    onda = np.sin(2 * np.pi * frequencia * projecao)
    return onda  # valores em [-1, 1]


def _deslocar_canal(canal: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(canal, dy, axis=0), dx, axis=1)


def aplicar(imagem_bgr: np.ndarray, rng: random.Random, intensidade: float = 1.0) -> tuple[np.ndarray, dict]:
    altura, largura = imagem_bgr.shape[:2]
    imagem_float = imagem_bgr.astype(np.float32)

    # 1. Moiré: soma uma modulação de luminância de alta frequência e baixa amplitude
    amplitude_moire = 8.0 * intensidade
    moire = _padrao_moire(altura, largura, rng)[..., None] * amplitude_moire
    imagem_float = imagem_float + moire

    # 2. Desalinhamento de canal (aberração cromática leve)
    b, g, r = cv2.split(imagem_float)
    dx, dy = rng.choice([-2, -1, 1, 2]), rng.choice([-1, 0, 1])
    r = _deslocar_canal(r, dx, dy)
    b = _deslocar_canal(b, -dx, -dy)
    imagem_float = cv2.merge([b, g, r])

    # 3. Reflexo especular parcial: gradiente diagonal suave somado como brilho extra
    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    angulo_reflexo = rng.uniform(0, 2 * np.pi)
    gradiente = (x * np.cos(angulo_reflexo) + y * np.sin(angulo_reflexo))
    gradiente = (gradiente - gradiente.min()) / (gradiente.max() - gradiente.min() + 1e-6)
    intensidade_reflexo = rng.uniform(5, 15) * intensidade
    imagem_float = imagem_float + (gradiente[..., None] * intensidade_reflexo)

    # 4. Leve desfoque (refotografar uma superfície emissora através do ar)
    imagem_uint8 = np.clip(imagem_float, 0, 255).astype(np.uint8)
    imagem_uint8 = cv2.GaussianBlur(imagem_uint8, (3, 3), sigmaX=0.6)

    parametros = {
        "amplitude_moire": amplitude_moire,
        "deslocamento_canal_px": [int(dx), int(dy)],
        "intensidade_reflexo": float(intensidade_reflexo),
        "aviso": "aproximacao_sintetica_nao_substitui_recaptura_real",
    }
    return imagem_uint8, parametros


def processar_documento(caminho: Path, tipo_documento: str, pasta_saida: Path, rng: random.Random, variantes: int) -> int:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0

    pasta_saida_tecnica = pasta_saida / "recaptura_tela"
    pasta_saida_tecnica.mkdir(parents=True, exist_ok=True)

    total = 0
    for i in range(variantes):
        intensidade = rng.uniform(0.6, 1.4)
        resultado, parametros = aplicar(imagem, rng, intensidade)
        nome_arquivo = f"{caminho.stem}__recaptura_tela_{i}.jpg"
        cv2.imwrite(str(pasta_saida_tecnica / nome_arquivo), resultado, [cv2.IMWRITE_JPEG_QUALITY, 85])

        registro = RegistroFraude(
            tecnica="recaptura_tela",
            documento_origem=str(caminho),
            tipo_documento=tipo_documento,
            arquivo_gerado=nome_arquivo,
            parametros=parametros,
            campo_alterado=None,  # afeta o documento inteiro, não um campo específico
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
    print(f"recaptura_tela: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'recaptura_tela'}")


if __name__ == "__main__":
    main()
