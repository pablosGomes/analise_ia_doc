"""Orquestrador — roda as 6 técnicas de geração de fraude sobre todos os
documentos legítimos e imprime um resumo final.

Uso:
    python3 -m scripts.geracao_fraude.gerar_dataset_fraude \
        --legitimos datasets/legitimos --saida datasets/fraude_gerada \
        --variantes-por-documento 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts.geracao_fraude import (
    crop_and_move,
    digito_verificador,
    edicao_campos,
    recaptura_tela,
    reimpressao,
    troca_foto,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument("--saida", default="datasets/fraude_gerada")
    parser.add_argument("--variantes-por-documento", type=int, default=10)
    parser.add_argument(
        "--tecnicas",
        nargs="+",
        default=["troca_foto", "crop_and_move", "digito_verificador", "edicao_campos", "recaptura_tela", "reimpressao"],
        help="Subconjunto de técnicas a rodar (por padrão, todas)",
    )
    args = parser.parse_args()

    pasta_legitimos = Path(args.legitimos)
    pasta_saida = Path(args.saida)

    if not pasta_legitimos.exists():
        raise SystemExit(f"Pasta de legítimos não encontrada: {pasta_legitimos}")

    total_documentos = sum(
        1 for p in pasta_legitimos.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if total_documentos == 0:
        raise SystemExit(f"Nenhum documento legítimo encontrado em {pasta_legitimos}")

    print(f"Documentos legítimos encontrados: {total_documentos}")
    print(f"Técnicas a rodar: {', '.join(args.tecnicas)}")
    print("-" * 60)

    resumo: dict[str, int] = {}
    inicio_total = time.time()

    if "troca_foto" in args.tecnicas:
        inicio = time.time()
        resumo["troca_foto"] = troca_foto.processar_dataset(pasta_legitimos, pasta_saida, args.variantes_por_documento)
        print(f"[1/6] troca_foto: {resumo['troca_foto']} imagens ({time.time() - inicio:.1f}s)")

    if "crop_and_move" in args.tecnicas:
        inicio = time.time()
        resumo["crop_and_move"] = crop_and_move.processar_dataset(pasta_legitimos, pasta_saida, args.variantes_por_documento)
        print(f"[2/6] crop_and_move: {resumo['crop_and_move']} imagens ({time.time() - inicio:.1f}s)")

    if "digito_verificador" in args.tecnicas:
        inicio = time.time()
        resumo["digito_verificador"] = digito_verificador.processar_dataset(pasta_legitimos, pasta_saida)
        print(f"[3/6] digito_verificador: {resumo['digito_verificador']} imagens ({time.time() - inicio:.1f}s)")

    if "edicao_campos" in args.tecnicas:
        inicio = time.time()
        resumo["edicao_campos"] = edicao_campos.processar_dataset(pasta_legitimos, pasta_saida)
        print(f"[4/6] edicao_campos: {resumo['edicao_campos']} imagens ({time.time() - inicio:.1f}s)")

    if "recaptura_tela" in args.tecnicas:
        inicio = time.time()
        resumo["recaptura_tela"] = recaptura_tela.processar_dataset(pasta_legitimos, pasta_saida, args.variantes_por_documento)
        print(f"[5/6] recaptura_tela: {resumo['recaptura_tela']} imagens ({time.time() - inicio:.1f}s)")

    if "reimpressao" in args.tecnicas:
        inicio = time.time()
        resumo["reimpressao"] = reimpressao.processar_dataset(pasta_legitimos, pasta_saida, args.variantes_por_documento)
        print(f"[6/6] reimpressao: {resumo['reimpressao']} imagens ({time.time() - inicio:.1f}s)")

    print("-" * 60)
    total_gerado = sum(resumo.values())
    print(f"Total: {total_gerado} imagens de fraude geradas em {time.time() - inicio_total:.1f}s")
    print(f"Saída: {pasta_saida.resolve()}")
    for tecnica, quantidade in resumo.items():
        print(f"  - {tecnica}: {quantidade}")


if __name__ == "__main__":
    main()
