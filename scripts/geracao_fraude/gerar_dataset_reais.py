"""Gerador de dataset a partir dos documentos REAIS (com rosto).

Gera as duas técnicas de fraude que fraudadores de fato usam — troca de foto de
rosto e edição de campos — e a classe LEGÍTIMA (os mesmos documentos pela captura
compartilhada), garantindo paridade de pipeline.

Os documentos reais NUNCA saem da máquina e nunca são versionados. Uso:

    python -m scripts.geracao_fraude.gerar_dataset_reais \
        --legitimos datasets/legitimos --saida datasets/gerado/reais_fraude \
        --legitimos-processados datasets/gerado/reais_legitimos \
        --variantes-por-documento 10
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2

from scripts.comum import valores
from scripts.geracao_fraude import edicao_campos, saida, troca_foto
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def processar_legitimos(pasta_legitimos, pasta_saida, variantes_por_documento, semente=7):
    """Passa cada documento legítimo pelo simulador de captura compartilhado,
    gerando a classe negativa (rotulo=legitimo) com a MESMA distribuição global
    das fraudes."""
    rng = random.Random(semente)
    total = 0
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        for caminho in sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM):
            imagem = cv2.imread(str(caminho))
            if imagem is None:
                continue
            for i in range(variantes_por_documento):
                reg = RegistroFraude(
                    tecnica="legitimo_processado", documento_origem=str(caminho),
                    tipo_documento=pasta_tipo.name, arquivo_gerado="", campo_alterado=None,
                    dificuldade="na", rotulo="legitimo",
                )
                saida.finalizar(imagem, rng, pasta_saida / pasta_tipo.name,
                                f"{caminho.stem}__legitimo_{i}", reg)
                total += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/gerado/reais_fraude")
    parser.add_argument("--legitimos-processados", default="datasets/gerado/reais_legitimos")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    parser.add_argument("--tecnicas", nargs="+", default=["troca_foto", "edicao_campos"])
    parser.add_argument("--manter-antigas", action="store_true",
                        help="não apagar a leva anterior (por padrão ela é apagada)")
    args = parser.parse_args()

    pl = Path(args.legitimos); ps = Path(args.saida); plp = Path(args.legitimos_processados)
    if not pl.exists():
        raise SystemExit(f"Pasta de legítimos não encontrada: {pl}")

    # Cada nova leva SUBSTITUI a anterior (não acumular imagens antigas em disco).
    if not args.manter_antigas:
        import shutil
        for raiz in (ps, plp):
            if raiz.exists():
                shutil.rmtree(raiz)
        print(f"Levas anteriores apagadas: {ps}, {plp}")
    valores.semear(42)
    print(f"Técnicas: {', '.join(args.tecnicas)}")
    print("-" * 60)

    resumo = {}
    t0 = time.time()
    if "troca_foto" in args.tecnicas:
        resumo["troca_foto"] = troca_foto.processar_dataset(pl, ps, args.variantes_por_documento)
    if "edicao_campos" in args.tecnicas:
        resumo["edicao_campos"] = edicao_campos.processar_dataset(pl, ps)

    resumo["legitimos_processados"] = processar_legitimos(pl, plp, args.variantes_por_documento)

    print("-" * 60)
    fraude = sum(v for k, v in resumo.items() if k != "legitimos_processados")
    print(f"Fraude: {fraude} | Legítimos processados: {resumo['legitimos_processados']} "
          f"| Total: {sum(resumo.values())} em {time.time() - t0:.1f}s")
    for k, v in resumo.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
