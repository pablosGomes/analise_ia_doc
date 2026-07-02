"""Técnica 1 — Substituição de foto (splicing de rosto).

Cola a foto de rosto de um documento sobre a foto de rosto de outro
documento, simulando a fraude mais comum em documentos de identidade:
substituir a foto do titular mantendo os demais campos.

Duas variantes de dificuldade, deliberadamente geradas em pares (o
classificador deve aprender a detectar as duas, não só a mais óbvia):

    - "evidente" (colagem_simples): recorte colado diretamente, sem
      suavização de borda nem correção de iluminação — produz uma
      descontinuidade de textura/ruído de sensor bem marcada, o tipo de
      fraude mais fácil de pegar.
    - "sutil" (blend_poisson): usa `cv2.seamlessClone` (Poisson blending),
      que ajusta gradientes de cor/iluminação na borda para uma transição
      visualmente suave — simula uma fraude "bem feita", que é o caso mais
      importante para o classificador aprender, já que uma colagem óbvia já
      seria pega por inspeção visual simples.

Fonte dos rostos substitutos: exclusivamente outros documentos do próprio
dataset `datasets/legitimos/` (nunca uma fonte externa) — o objetivo é gerar
uma fraude *plausível* usando dados já cobertos pelo consentimento obtido,
não recriar uma pessoa real que não deu consentimento para a variante
gerada.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.comum import deteccao
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def _rosto_principal(imagem_bgr: np.ndarray) -> deteccao.CaixaDelimitadora | None:
    return deteccao.maior_caixa(deteccao.detectar_rostos(imagem_bgr))


def aplicar_colagem_simples(
    imagem_alvo: np.ndarray, recorte_rosto: np.ndarray, caixa_alvo: deteccao.CaixaDelimitadora
) -> np.ndarray:
    resultado = imagem_alvo.copy()
    rosto_redimensionado = cv2.resize(recorte_rosto, (caixa_alvo.w, caixa_alvo.h), interpolation=cv2.INTER_LINEAR)
    resultado[caixa_alvo.y:caixa_alvo.y2, caixa_alvo.x:caixa_alvo.x2] = rosto_redimensionado
    return resultado


def aplicar_blend_poisson(
    imagem_alvo: np.ndarray, recorte_rosto: np.ndarray, caixa_alvo: deteccao.CaixaDelimitadora
) -> np.ndarray:
    rosto_redimensionado = cv2.resize(recorte_rosto, (caixa_alvo.w, caixa_alvo.h), interpolation=cv2.INTER_LINEAR)

    mascara = np.full((caixa_alvo.h, caixa_alvo.w), 255, dtype=np.uint8)
    # Encolhe a máscara alguns pixels da borda: cv2.seamlessClone exige que a
    # máscara não toque a borda do recorte fonte, senão lança erro interno.
    encolhimento = max(1, min(caixa_alvo.w, caixa_alvo.h) // 20)
    mascara = cv2.erode(mascara, np.ones((encolhimento, encolhimento), np.uint8))

    centro = (caixa_alvo.x + caixa_alvo.w // 2, caixa_alvo.y + caixa_alvo.h // 2)
    try:
        resultado = cv2.seamlessClone(rosto_redimensionado, imagem_alvo, mascara, centro, cv2.NORMAL_CLONE)
    except cv2.error:
        # Poisson blending pode falhar em recortes muito pequenos/degenerados;
        # cai para colagem simples nesse caso (ainda é uma variante válida,
        # só que registrada com o modo efetivamente usado).
        return aplicar_colagem_simples(imagem_alvo, recorte_rosto, caixa_alvo)
    return resultado


MODOS = {
    "colagem_simples": (aplicar_colagem_simples, "evidente"),
    "blend_poisson": (aplicar_blend_poisson, "sutil"),
}


def gerar_variantes_para_par(
    caminho_alvo: Path, caminho_fonte_rosto: Path, tipo_documento: str
) -> list[tuple[np.ndarray, RegistroFraude]]:
    """Gera as variantes (uma por modo em MODOS) trocando o rosto de
    `caminho_alvo` pelo rosto extraído de `caminho_fonte_rosto`."""
    imagem_alvo = cv2.imread(str(caminho_alvo))
    imagem_fonte = cv2.imread(str(caminho_fonte_rosto))
    if imagem_alvo is None or imagem_fonte is None:
        return []

    caixa_alvo = _rosto_principal(imagem_alvo)
    caixa_fonte = _rosto_principal(imagem_fonte)
    if caixa_alvo is None or caixa_fonte is None:
        return []

    recorte_rosto = imagem_fonte[caixa_fonte.y:caixa_fonte.y2, caixa_fonte.x:caixa_fonte.x2]
    if recorte_rosto.size == 0:
        return []

    resultados = []
    for nome_modo, (funcao, dificuldade) in MODOS.items():
        imagem_resultado = funcao(imagem_alvo, recorte_rosto, caixa_alvo)
        registro = RegistroFraude(
            tecnica="troca_foto",
            documento_origem=str(caminho_alvo),
            tipo_documento=tipo_documento,
            arquivo_gerado="",  # preenchido pelo chamador, que sabe o nome final do arquivo
            parametros={
                "modo": nome_modo,
                "fonte_rosto": str(caminho_fonte_rosto),
                "caixa_alvo": caixa_alvo.como_tupla(),
            },
            campo_alterado="foto_rosto",
            dificuldade=dificuldade,
        )
        resultados.append((imagem_resultado, registro))
    return resultados


def processar_dataset(
    pasta_legitimos: Path, pasta_saida: Path, variantes_por_documento: int, semente: int = 42
) -> int:
    """Para cada documento legítimo, seleciona outros documentos do dataset
    (de preferência do mesmo tipo, para manter o recorte de rosto com
    proporções plausíveis) como fonte de rosto substituto, e gera as
    variantes. Retorna o total de imagens geradas."""
    random.seed(semente)
    arquivos_por_tipo: dict[str, list[Path]] = {}
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        arquivos = sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM)
        if arquivos:
            arquivos_por_tipo[pasta_tipo.name] = arquivos

    total_gerado = 0
    for tipo_documento, arquivos in arquivos_por_tipo.items():
        pasta_saida_tipo = pasta_saida / "troca_foto"
        for caminho_alvo in arquivos:
            # Pool de fontes de rosto: outros arquivos do mesmo tipo primeiro
            # (proporção de rosto mais parecida); se não houver outro do
            # mesmo tipo, usa qualquer outro documento do dataset.
            candidatos = [a for a in arquivos if a != caminho_alvo]
            if not candidatos:
                candidatos = [
                    a for lista in arquivos_por_tipo.values() for a in lista if a != caminho_alvo
                ]
            if not candidatos:
                continue

            random.shuffle(candidatos)
            pares_gerados = 0
            for caminho_fonte in itertools.cycle(candidatos):
                if pares_gerados * len(MODOS) >= variantes_por_documento:
                    break
                variantes = gerar_variantes_para_par(caminho_alvo, caminho_fonte, tipo_documento)
                for i, (imagem_resultado, registro) in enumerate(variantes):
                    nome_arquivo = f"{caminho_alvo.stem}__troca_foto_{registro.parametros['modo']}_{pares_gerados}.jpg"
                    registro.arquivo_gerado = nome_arquivo
                    pasta_saida_tipo.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pasta_saida_tipo / nome_arquivo), imagem_resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    registro.salvar(pasta_saida_tipo)
                    total_gerado += 1
                pares_gerados += 1
                if len(candidatos) == 1 and pares_gerados >= 1:
                    # Evita loop infinito quando só existe 1 candidato possível
                    # e ele já não rendeu variante suficiente (ex.: sem rosto detectado).
                    if pares_gerados * len(MODOS) < variantes_por_documento and pares_gerados > 20:
                        break
    return total_gerado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos", help="Pasta com subpastas rg/cnh/passaporte")
    parser.add_argument("--saida", default="datasets/fraude_gerada", help="Pasta base de saída")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"troca_foto: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'troca_foto'}")


if __name__ == "__main__":
    main()
