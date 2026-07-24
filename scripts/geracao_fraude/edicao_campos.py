"""Técnica 4 — Edição de campos de texto.

Edita nome, nascimento, filiação ou assinatura. Os valores vêm do Faker pt-BR e a
fonte de um pool amplo; o texto original é removido por inpainting. As variantes
"fonte_correta"/"fonte_incorreta" usam famílias de fonte diferentes mas plausíveis.
A saída passa pelo simulador de captura compartilhado.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from scripts.comum import deteccao, inpaint, valores
from scripts.comum.renderizacao_texto import desenhar_texto, estimar_cor_tinta, nitidez_regiao, extensao_texto
from scripts.comum.textura import harmonizar_textura
from scripts.geracao_fraude import saida
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

PALAVRAS_CHAVE_CAMPOS = {
    "nome": ["NOME"],
    "nascimento": ["NASCIMENTO"],
    "filiacao": ["FILIACAO", "FILIAÇÃO", "MAE", "MÃE", "PAI"],
    "assinatura": ["ASSINATURA"],
}


def _valor_para_campo(nome_campo, rng):
    if nome_campo == "nome":
        return valores.nome(rng)
    if nome_campo == "nascimento":
        return valores.data_nascimento(rng)
    if nome_campo == "filiacao":
        return valores.filiacao(rng)
    return None


def _rabisco_assinatura(imagem_bgr, x, y, w, h, rng, cor_bgr=(20, 20, 20)):
    r = imagem_bgr.copy()
    n = rng.randint(6, 10)
    pontos = [(x + int(w * i / (n - 1)), y + h // 2 + rng.randint(-h // 3, h // 3)) for i in range(n)]
    esp = max(1, h // 12)
    for i in range(len(pontos) - 1):
        cv2.line(r, pontos[i], pontos[i + 1], cor_bgr, thickness=max(1, esp + rng.randint(-1, 1)), lineType=cv2.LINE_AA)
    return r


def processar_documento(caminho, tipo_documento, pasta_saida, rng):
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0
    pasta = pasta_saida / "edicao_campos"
    total = 0
    for nome_campo, palavras in PALAVRAS_CHAVE_CAMPOS.items():
        caixas = deteccao.detectar_valores_proximos_a_rotulo(imagem, palavras)
        if not caixas:
            continue
        mesma_linha = [c for c in caixas if c.motivo.startswith("valor_apos_rotulo")]
        caixa = deteccao.maior_caixa(mesma_linha) or deteccao.maior_caixa(caixas)
        # Alarga a caixa para cobrir todo o texto original (evita fantasma do original
        # quando o valor detectado é mais estreito que o texto de fato).
        if nome_campo == "assinatura":
            bx, by, bw, bh = caixa.x, caixa.y, caixa.w, caixa.h
        else:
            bx, by, bw, bh = extensao_texto(imagem, caixa.x, caixa.y, caixa.w, caixa.h)
        cor = estimar_cor_tinta(imagem, bx, by, bw, bh)
        nit = nitidez_regiao(imagem, bx, by, bw, bh)

        for fonte_correta in (True, False):
            sem_texto, metodo_inpaint = inpaint.remover_tinta(imagem, bx, by, bw, bh, rng, margem=6)
            if nome_campo == "assinatura":
                res = _rabisco_assinatura(sem_texto, bx, by, bw, bh, rng, cor_bgr=cor)
                res = harmonizar_textura(res, bx, by, bw, bh, rng)
                valor, fonte = "rabisco_sintetico", None
            else:
                valor = _valor_para_campo(nome_campo, rng)
                fonte = valores.escolher_fonte(rng, correta=fonte_correta)
                res = desenhar_texto(sem_texto, valor, bx, by, bw, bh,
                                     cor_bgr=cor, caminho_fonte=fonte, rng=rng, nitidez_alvo=nit)
            sufixo = "fonte_correta" if fonte_correta else "fonte_incorreta"
            reg = RegistroFraude(
                tecnica="edicao_campos", documento_origem=str(caminho), tipo_documento=tipo_documento,
                arquivo_gerado="", campo_alterado=nome_campo,
                dificuldade="sutil" if fonte_correta else "evidente",
                parametros={"caixa": caixa.como_tupla(), "valor_substituto": valor, "fonte_correta": fonte_correta},
                metodo_inpaint=metodo_inpaint, fonte=(Path(fonte).name if fonte else None),
                gerador="render_classico",
            )
            saida.finalizar(res, rng, pasta, f"{caminho.stem}__edicao_campos_{nome_campo}_{sufixo}", reg)
            total += 1
    return total


def processar_dataset(pasta_legitimos, pasta_saida, semente=42):
    rng = random.Random(semente); valores.semear(semente); total = 0
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        for caminho in sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM):
            total += processar_documento(caminho, pasta_tipo.name, pasta_saida, rng)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/gerado/reais_fraude")
    args = parser.parse_args()
    total = processar_dataset(Path(args.legitimos), Path(args.saida))
    print(f"edicao_campos: {total} imagem(ns) gerada(s)")


if __name__ == "__main__":
    main()
