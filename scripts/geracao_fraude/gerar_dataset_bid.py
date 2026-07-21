"""Gerador de dataset a partir do BID Dataset (documentos brasileiros sintéticos).

Para cada documento amostrado, gera a fraude de EDIÇÃO DE CAMPOS (reescreve
nome/filiação/data/CPF/registro usando as caixas ground-truth do BID) e também a
classe LEGÍTIMA (o mesmo documento pela captura compartilhada). A paridade de
pipeline garante que a única diferença sistemática entre as classes seja a
manipulação local.

A troca de foto NÃO é gerada aqui: no BID a foto de rosto é um placeholder cinza,
então face swap não produz amostra útil — essa técnica vem dos documentos reais
(`gerar_dataset_reais.py`).

Saída:
    datasets/gerado/bid_fraude/edicao_campos/
    datasets/gerado/bid_legitimos/<classe>/

Uso:
    python -m scripts.geracao_fraude.gerar_dataset_bid --por-classe 150 --variantes-legit 7
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from scripts.comum import bid, inpaint, valores
from scripts.comum.renderizacao_texto import desenhar_texto, estimar_cor_tinta, nitidez_regiao, extensao_texto
from scripts.geracao_fraude import saida
from scripts.geracao_fraude.manifesto import RegistroFraude


def _doc_id(caminho):
    return caminho.name.replace("_in.jpg", "")


def _valor_substituto(campo, rng):
    if campo.tipo == "nome":
        return valores.nome(rng)
    if campo.tipo == "filiacao":
        return valores.filiacao(rng)
    if campo.tipo == "data":
        return valores.data_nascimento(rng)
    if campo.tipo == "cpf":
        d = valores.cpf_valido(rng)
        return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"
    n = len([c for c in campo.texto_original if c.isdigit()]) or 9
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _editar_campos(doc, caminho, tipo_doc, pasta, rng, n):
    """Reescreve até `n` campos do documento, gerando `n` amostras de fraude."""
    imagem, campos = doc["imagem"], doc["campos"]
    escolhidos = campos[:]
    rng.shuffle(escolhidos)
    total = 0
    for i, campo in enumerate(escolhidos[:n]):
        fonte_correta = rng.random() < 0.5
        # Alarga a caixa para cobrir todo o texto original (evita fantasma do original).
        bx, by, bw, bh = extensao_texto(imagem, campo.x, campo.y, campo.w, campo.h)
        cor = estimar_cor_tinta(imagem, bx, by, bw, bh)
        nit = nitidez_regiao(imagem, bx, by, bw, bh)
        sem_texto, metodo = inpaint.remover_tinta(imagem, bx, by, bw, bh, rng, margem=6)
        valor = _valor_substituto(campo, rng)
        fonte = valores.escolher_fonte(rng, correta=fonte_correta)
        res = desenhar_texto(sem_texto, valor, bx, by, bw, bh,
                             cor_bgr=cor, caminho_fonte=fonte, rng=rng, nitidez_alvo=nit)
        suf = "fonte_correta" if fonte_correta else "fonte_incorreta"
        reg = RegistroFraude(
            tecnica="edicao_campos", documento_origem=str(caminho), tipo_documento=tipo_doc,
            arquivo_gerado="", campo_alterado=campo.tipo,
            dificuldade="sutil" if fonte_correta else "evidente", rotulo="fraude",
            parametros={"origem": "bid", "caixa": campo.como_tupla(), "valor_substituto": valor,
                        "texto_original": campo.texto_original, "fonte_correta": fonte_correta,
                        "orientacao": doc["rotacao"]},
            metodo_inpaint=metodo, fonte=Path(fonte).name)
        saida.finalizar(res, rng, pasta / "edicao_campos",
                        f"{_doc_id(caminho)}__{tipo_doc}__edicao_{i}_{campo.tipo}_{suf}", reg)
        total += 1
    return total


def _gerar_legitimos(doc, caminho, tipo_doc, pasta_legit, rng, variantes):
    total = 0
    for i in range(variantes):
        reg = RegistroFraude(
            tecnica="legitimo_processado", documento_origem=str(caminho), tipo_documento=tipo_doc,
            arquivo_gerado="", campo_alterado=None, dificuldade="na", rotulo="legitimo",
            parametros={"origem": "bid", "orientacao": doc["rotacao"]})
        saida.finalizar(doc["imagem"], rng, pasta_legit / tipo_doc,
                        f"{_doc_id(caminho)}__{tipo_doc}__legitimo_{i}", reg)
        total += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bid", default="datasets/BID Dataset")
    parser.add_argument("--fraude", default="datasets/gerado/bid_fraude")
    parser.add_argument("--legitimos-processados", default="datasets/gerado/bid_legitimos")
    parser.add_argument("--classes", nargs="+", default=list(bid.CLASSES_RICAS))
    parser.add_argument("--por-classe", type=int, default=150)
    parser.add_argument("--edicao-por-doc", type=int, default=2)
    parser.add_argument("--variantes-legit", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manter-antigas", action="store_true",
                        help="não apagar a leva anterior (por padrão ela é apagada)")
    args = parser.parse_args()

    pasta_bid = Path(args.bid)
    if not pasta_bid.exists():
        raise SystemExit(f"Pasta do BID não encontrada: {pasta_bid}")
    pf = Path(args.fraude); pl = Path(args.legitimos_processados)

    # Cada nova leva SUBSTITUI a anterior (não acumular imagens antigas em disco).
    if not args.manter_antigas:
        import shutil
        for raiz in (pf, pl):
            if raiz.exists():
                shutil.rmtree(raiz)
        print(f"Levas anteriores apagadas: {pf}, {pl}")

    rng = random.Random(args.seed)
    valores.semear(args.seed)
    amostra = bid.amostrar(pasta_bid, args.classes, args.por_classe, rng)
    print(f"Classes: {', '.join(args.classes)} | amostrados: {len(amostra)} docs")
    print("-" * 62)

    t0 = time.time()
    n_fraude = n_legit = n_ok = n_desc = 0
    for idx, (caminho, tipo_doc) in enumerate(amostra, 1):
        doc = bid.carregar_documento(caminho)
        if doc is None:
            n_desc += 1
            continue
        n_ok += 1
        n_fraude += _editar_campos(doc, caminho, tipo_doc, pf, rng, args.edicao_por_doc)
        n_legit += _gerar_legitimos(doc, caminho, tipo_doc, pl, rng, args.variantes_legit)
        if idx % 50 == 0:
            print(f"  {idx}/{len(amostra)} docs | fraude={n_fraude} legit={n_legit} desc={n_desc} | {time.time()-t0:.0f}s")

    print("-" * 62)
    print(f"Docs usados: {n_ok} | descartados: {n_desc}")
    print(f"Edição de campos: {n_fraude} | Legítimos: {n_legit} | Total: {n_fraude + n_legit} em {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
