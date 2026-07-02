"""Técnica 2 — Crop-and-move.

Recorta uma região retangular do próprio documento e a reposiciona em outro
ponto, sobrescrevendo o conteúdo original ali — a técnica usada pelo dataset
acadêmico SIDTD (sobre a base MIDV-2020) para gerar fraudes a partir de
documentos legítimos, e que valida diretamente a estratégia deste projeto.

Duas variantes:
    - "duplicacao": copia uma região para outro ponto sem removê-la da
      origem (ex.: duplicar um selo/carimbo em outro lugar do documento).
    - "deslocamento": move a região (remove da origem preenchendo com a cor
      mediana local, cola no destino) — simula, por exemplo, deslocar um
      número ou data para encobrir uma rasura.

A região é escolhida aleatoriamente entre 5% e 15% da largura/altura do
documento, evitando a área do rosto principal (para não se sobrepor
acidentalmente com a técnica de troca de foto e produzir uma variante
ambígua).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.comum import deteccao
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def _preencher_com_mediana_local(imagem: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    alt, larg = imagem.shape[:2]
    margem = max(w, h)
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(larg, x + w + margem), min(alt, y + h + margem)
    vizinhanca = imagem[y0:y1, x0:x1].reshape(-1, imagem.shape[-1])
    mediana = np.median(vizinhanca, axis=0)
    ruido = np.random.normal(0, 5, (h, w, imagem.shape[-1])).astype(np.int16)
    imagem[y:y + h, x:x + w] = np.clip(mediana.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def _sobrepoe(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _escolher_regiao(
    largura: int, altura: int, rng: random.Random, evitar: list[tuple[int, int, int, int]], tentativas: int = 30
) -> tuple[int, int, int, int] | None:
    for _ in range(tentativas):
        w = rng.randint(int(0.05 * largura), int(0.15 * largura))
        h = rng.randint(int(0.05 * altura), int(0.15 * altura))
        x = rng.randint(0, max(0, largura - w))
        y = rng.randint(0, max(0, altura - h))
        candidata = (x, y, w, h)
        if not any(_sobrepoe(candidata, e) for e in evitar):
            return candidata
    return None


def aplicar(
    imagem_bgr: np.ndarray, modo: str, rng: random.Random, evitar: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, dict] | None:
    altura, largura = imagem_bgr.shape[:2]
    origem = _escolher_regiao(largura, altura, rng, evitar)
    if origem is None:
        return None
    destino = _escolher_regiao(largura, altura, rng, evitar + [origem])
    if destino is None:
        return None

    ox, oy, ow, oh = origem
    dx, dy, dw, dh = destino
    resultado = imagem_bgr.copy()
    recorte = imagem_bgr[oy:oy + oh, ox:ox + ow]
    recorte_redimensionado = cv2.resize(recorte, (dw, dh), interpolation=cv2.INTER_LINEAR)
    resultado[dy:dy + dh, dx:dx + dw] = recorte_redimensionado

    if modo == "deslocamento":
        _preencher_com_mediana_local(resultado, ox, oy, ow, oh)

    parametros = {"modo": modo, "origem": origem, "destino": destino}
    return resultado, parametros


def processar_documento(caminho: Path, tipo_documento: str, pasta_saida: Path, variantes: int, rng: random.Random) -> int:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0

    caixa_rosto = deteccao.maior_caixa(deteccao.detectar_rostos(imagem))
    evitar = [caixa_rosto.como_tupla()] if caixa_rosto is not None else []

    pasta_saida_tecnica = pasta_saida / "crop_and_move"
    pasta_saida_tecnica.mkdir(parents=True, exist_ok=True)

    modos = ["duplicacao", "deslocamento"]
    total = 0
    for i in range(variantes):
        modo = modos[i % len(modos)]
        resultado = aplicar(imagem, modo, rng, evitar)
        if resultado is None:
            continue
        imagem_resultado, parametros = resultado
        nome_arquivo = f"{caminho.stem}__crop_and_move_{modo}_{i}.jpg"
        cv2.imwrite(str(pasta_saida_tecnica / nome_arquivo), imagem_resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])

        registro = RegistroFraude(
            tecnica="crop_and_move",
            documento_origem=str(caminho),
            tipo_documento=tipo_documento,
            arquivo_gerado=nome_arquivo,
            parametros=parametros,
            campo_alterado="regiao_arbitraria",
            dificuldade="sutil" if modo == "deslocamento" else "media",
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
            total += processar_documento(caminho, pasta_tipo.name, pasta_saida, variantes_por_documento, rng)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/fraude_gerada")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"crop_and_move: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'crop_and_move'}")


if __name__ == "__main__":
    main()
