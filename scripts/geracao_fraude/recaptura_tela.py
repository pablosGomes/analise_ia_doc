"""Técnica 5 — Recaptura de tela (screen/replay attack).

LIMITAÇÃO IMPORTANTE: aproximação sintética computacional do artefato real
de recaptura de tela (moiré, reflexo especular, padrão de subpixel).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.comum.ruido import cascata_recompressao_jpeg, moire_por_subamostragem, ruido_sensor_poisson_gaussiano
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def _deslocar_canal(canal, dx, dy):
    return np.roll(np.roll(canal, dy, axis=0), dx, axis=1)


def aplicar(imagem_bgr, rng, intensidade=1.0):
    altura, largura = imagem_bgr.shape[:2]
    imagem_float = imagem_bgr.astype(np.float32)

    amplitude_moire = 8.0 * intensidade
    moire = moire_por_subamostragem(altura, largura, rng)[..., None] * amplitude_moire
    imagem_float = imagem_float + moire

    b, g, r = cv2.split(imagem_float)
    dx, dy = rng.choice([-2, -1, 1, 2]), rng.choice([-1, 0, 1])
    r = _deslocar_canal(r, dx, dy)
    b = _deslocar_canal(b, -dx, -dy)
    imagem_float = cv2.merge([b, g, r])

    y, x = np.mgrid[0:altura, 0:largura].astype(np.float32)
    angulo_reflexo = rng.uniform(0, 2 * np.pi)
    gradiente = (x * np.cos(angulo_reflexo) + y * np.sin(angulo_reflexo))
    gradiente = (gradiente - gradiente.min()) / (gradiente.max() - gradiente.min() + 1e-6)
    intensidade_reflexo = rng.uniform(5, 15) * intensidade
    imagem_float = imagem_float + (gradiente[..., None] * intensidade_reflexo)

    ganho_iso = rng.uniform(1.2, 3.0) * intensidade
    imagem_float = ruido_sensor_poisson_gaussiano(imagem_float, ganho_iso, rng)

    imagem_uint8 = np.clip(imagem_float, 0, 255).astype(np.uint8)
    imagem_uint8 = cv2.GaussianBlur(imagem_uint8, (3, 3), sigmaX=0.6)

    qualidades_cascata = [rng.randint(80, 95), rng.randint(70, 88)]
    imagem_uint8 = cascata_recompressao_jpeg(imagem_uint8, qualidades_cascata)

    parametros = {
        "amplitude_moire": amplitude_moire,
        "deslocamento_canal_px": [int(dx), int(dy)],
        "intensidade_reflexo": float(intensidade_reflexo),
        "ganho_iso_ruido_sensor": float(ganho_iso),
        "qualidades_cascata_jpeg": qualidades_cascata,
        "aviso": "aproximacao_sintetica_nao_substitui_recaptura_real",
    }
    return imagem_uint8, parametros


def processar_documento(caminho, tipo_documento, pasta_saida, rng, variantes):
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
            campo_alterado=None,
            dificuldade="sutil",
        )
        registro.salvar(pasta_saida_tecnica)
        total += 1
    return total


def processar_dataset(pasta_legitimos, pasta_saida, variantes_por_documento, semente=42):
    rng = random.Random(semente)
    total = 0
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        for caminho in sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM):
            total += processar_documento(caminho, pasta_tipo.name, pasta_saida, rng, variantes_por_documento)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/fraude_gerada")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"recaptura_tela: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'recaptura_tela'}")


if __name__ == "__main__":
    main()
