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
    - "sutil" (blend_poisson): calibra a cor (transferência Reinhard em LAB,
      usando o anel de pixels ao redor da caixa do rosto como referência de
      iluminação ambiente/temperatura de cor do documento de destino) e a
      nitidez (equaliza a variância do Laplaciano do recorte à do entorno,
      simulando o descasamento de resolução/câmera típico de um recorte
      vindo de outra foto) do rosto substituto ANTES do Poisson blending —
      simula uma fraude "bem feita", que é o caso mais importante para o
      classificador aprender, já que uma colagem óbvia já seria pega por
      inspeção visual simples.

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


def _rosto_principal(imagem_bgr):
    return deteccao.maior_caixa(deteccao.detectar_rostos(imagem_bgr))


def _anel_ao_redor(imagem_bgr, caixa, margem_px):
    altura, largura = imagem_bgr.shape[:2]
    x1 = max(0, caixa.x - margem_px)
    y1 = max(0, caixa.y - margem_px)
    x2 = min(largura, caixa.x2 + margem_px)
    y2 = min(altura, caixa.y2 + margem_px)
    regiao_expandida = imagem_bgr[y1:y2, x1:x2].copy()

    mascara = np.ones(regiao_expandida.shape[:2], dtype=bool)
    ix1, iy1 = caixa.x - x1, caixa.y - y1
    ix2, iy2 = caixa.x2 - x1, caixa.y2 - y1
    mascara[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)] = False

    pixels_anel = regiao_expandida[mascara]
    return pixels_anel if pixels_anel.size else regiao_expandida.reshape(-1, 3)


def _estatisticas_lab(pixels_bgr):
    amostra = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(amostra, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab.mean(axis=0), lab.std(axis=0) + 1e-6


def _transferir_cor_lab(recorte_bgr, media_alvo, desvio_alvo):
    lab = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    media_origem = lab.reshape(-1, 3).mean(axis=0)
    desvio_origem = lab.reshape(-1, 3).std(axis=0) + 1e-6

    lab_ajustado = (lab - media_origem) * (desvio_alvo / desvio_origem) + media_alvo
    lab_ajustado = np.clip(lab_ajustado, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_ajustado, cv2.COLOR_LAB2BGR)


def _variancia_laplaciana(imagem_bgr):
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


def _igualar_nitidez(recorte_bgr, variancia_alvo, tentativas=6):
    variancia_recorte = _variancia_laplaciana(recorte_bgr)
    if variancia_recorte <= variancia_alvo * 1.15:
        return recorte_bgr

    resultado = recorte_bgr
    sigma = 0.4
    for _ in range(tentativas):
        candidato = cv2.GaussianBlur(recorte_bgr, (0, 0), sigmaX=sigma)
        if _variancia_laplaciana(candidato) <= variancia_alvo * 1.15:
            resultado = candidato
            break
        resultado = candidato
        sigma += 0.4
    return resultado


def aplicar_colagem_simples(imagem_alvo, recorte_rosto, caixa_alvo):
    resultado = imagem_alvo.copy()
    rosto_redimensionado = cv2.resize(recorte_rosto, (caixa_alvo.w, caixa_alvo.h), interpolation=cv2.INTER_LINEAR)
    resultado[caixa_alvo.y:caixa_alvo.y2, caixa_alvo.x:caixa_alvo.x2] = rosto_redimensionado
    return resultado


def aplicar_blend_poisson(imagem_alvo, recorte_rosto, caixa_alvo):
    rosto_redimensionado = cv2.resize(recorte_rosto, (caixa_alvo.w, caixa_alvo.h), interpolation=cv2.INTER_LINEAR)

    margem = max(4, min(caixa_alvo.w, caixa_alvo.h) // 6)
    pixels_anel = _anel_ao_redor(imagem_alvo, caixa_alvo, margem)
    media_alvo, desvio_alvo = _estatisticas_lab(pixels_anel)
    rosto_redimensionado = _transferir_cor_lab(rosto_redimensionado, media_alvo, desvio_alvo)

    recorte_original_alvo = imagem_alvo[caixa_alvo.y:caixa_alvo.y2, caixa_alvo.x:caixa_alvo.x2]
    variancia_referencia = (
        _variancia_laplaciana(recorte_original_alvo) if recorte_original_alvo.size else _variancia_laplaciana(imagem_alvo)
    )
    rosto_redimensionado = _igualar_nitidez(rosto_redimensionado, variancia_referencia)

    mascara = np.full((caixa_alvo.h, caixa_alvo.w), 255, dtype=np.uint8)
    encolhimento = max(1, min(caixa_alvo.w, caixa_alvo.h) // 20)
    mascara = cv2.erode(
        mascara, np.ones((encolhimento, encolhimento), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )

    centro = (caixa_alvo.x + caixa_alvo.w // 2, caixa_alvo.y + caixa_alvo.h // 2)
    try:
        resultado = cv2.seamlessClone(rosto_redimensionado, imagem_alvo, mascara, centro, cv2.NORMAL_CLONE)
    except cv2.error:
        return aplicar_colagem_simples(imagem_alvo, recorte_rosto, caixa_alvo)
    return resultado


MODOS = {
    "colagem_simples": (aplicar_colagem_simples, "evidente"),
    "blend_poisson": (aplicar_blend_poisson, "sutil"),
}


def gerar_variantes_para_par(caminho_alvo, caminho_fonte_rosto, tipo_documento):
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
            arquivo_gerado="",
            parametros={
                "modo": nome_modo,
                "fonte_rosto": str(caminho_fonte_rosto),
                "caixa_alvo": caixa_alvo.como_tupla(),
                "calibracao_cor_nitidez": nome_modo == "blend_poisson",
            },
            campo_alterado="foto_rosto",
            dificuldade=dificuldade,
        )
        resultados.append((imagem_resultado, registro))
    return resultados


def processar_dataset(pasta_legitimos, pasta_saida, variantes_por_documento, semente=42):
    random.seed(semente)
    arquivos_por_tipo = {}
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
                    if pares_gerados * len(MODOS) < variantes_por_documento and pares_gerados > 20:
                        break
    return total_gerado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos", help="Pasta com subpastas rg/cnh/passaporte")
    parser.add_argument("--saida", default="datasets/fraude_gerada", help="Pasta base de saída")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida), args.variantes_por_documento)
    print(f"troca_foto: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'troca_foto'}")


if __name__ == "__main__":
    main()
