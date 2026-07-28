"""Carrega para o MongoDB os metadados do dataset gerado e as métricas de treino.

Fontes:
    datasets/gerado/**/*.manifesto.json   -> coleção `amostras`
    datasets/processed/*.jsonl (treino)   -> coleções `metricas_treino` e `avaliacoes`

É idempotente: reprocessar as mesmas fontes atualiza os documentos em vez de
duplicar (upsert pelo `id` do manifesto / pela chave do evento).

NADA de imagem ou PII entra no banco: só metadados. O caminho do documento de
origem vira hash (ver `mongo.id_documento`); campos de texto livre que possam
carregar dado real são descartados nas amostras de origem `reais`.

Uso:
    python -m scripts.db.ingestao                      # ingere tudo
    python -m scripts.db.ingestao --resumo             # só mostra o que há no banco
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pymongo import UpdateOne

from scripts.db.mongo import conectar, garantir_indices, id_documento

# subpasta de datasets/gerado -> origem dos dados
GRUPOS = {"bid_fraude": "bid", "bid_legitimos": "bid",
          "reais_fraude": "reais", "reais_legitimos": "reais"}

# Campos de texto livre do manifesto que NÃO podem ir para o banco quando a
# amostra vem de documento real (poderiam conter dado pessoal transcrito).
CAMPOS_TEXTO_SENSIVEL = ("texto_original",)


_RE_CAMINHO = re.compile(r"[/\\]|\.(?:jpe?g|png|tif?f|bmp)$", re.I)


def _sanitizar_valor(v):
    """Troca qualquer string que pareça caminho/arquivo de imagem pelo hash dela.

    Alguns parâmetros guardam o caminho de um documento real (ex.: `fonte_rosto`,
    o documento doador do rosto na troca de foto). O hash mantém a rastreabilidade
    (dá para saber que duas amostras usaram o mesmo doador) sem gravar qual é.
    """
    if isinstance(v, str) and _RE_CAMINHO.search(v):
        return id_documento(v)
    if isinstance(v, dict):
        return {k: _sanitizar_valor(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_sanitizar_valor(x) for x in v]
    return v


def _sanitizar_arquivo(nome: str, doc_id: str) -> str:
    """Nome de arquivo seguro para gravar no banco.

    O nome gerado começa com o stem do documento de origem — nos documentos reais
    isso é o nome do arquivo original (ex.: 'WhatsApp Image 2026-07-02 at 11.13.00'),
    que identifica o documento e não pode ir para o banco. Troca esse prefixo pelo
    hash do documento, preservando o sufixo descritivo (técnica, campo, variante).
    """
    sufixo = nome.split("__", 1)[1] if "__" in nome else Path(nome).name
    return f"{doc_id}__{sufixo}"


def _amostra_do_manifesto(caminho: Path, fonte_dados: str) -> dict | None:
    try:
        m = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    parametros = dict(m.get("parametros") or {})
    if fonte_dados == "reais":
        for campo in CAMPOS_TEXTO_SENSIVEL:
            parametros.pop(campo, None)
        parametros = _sanitizar_valor(parametros)
    doc_id = id_documento(m.get("documento_origem", ""))
    arquivo = m.get("arquivo_gerado") or ""
    if fonte_dados == "reais":
        arquivo = _sanitizar_arquivo(arquivo, doc_id)
    return {
        "id": m.get("id"),
        "rotulo": m.get("rotulo", "fraude"),
        "tecnica": m.get("tecnica"),
        "gerador": m.get("gerador"),
        "tipo_documento": m.get("tipo_documento"),
        "dificuldade": m.get("dificuldade"),
        "campo_alterado": m.get("campo_alterado"),
        "metodo_inpaint": m.get("metodo_inpaint"),
        "fonte_tipografica": m.get("fonte"),          # arquivo .ttf usado no texto
        "fonte_dados": fonte_dados,                    # bid | reais
        "documento_id": doc_id,
        "arquivo": arquivo,
        "parametros": parametros,
        "parametros_captura": m.get("parametros_captura") or {},
        "timestamp": m.get("timestamp"),
    }


def ingerir_amostras(db, pasta_gerado: Path) -> int:
    """Varre os manifestos e faz upsert na coleção `amostras`."""
    operacoes = []
    for subpasta, fonte_dados in GRUPOS.items():
        raiz = pasta_gerado / subpasta
        if not raiz.exists():
            continue
        for caminho in raiz.rglob("*.manifesto.json"):
            doc = _amostra_do_manifesto(caminho, fonte_dados)
            if doc and doc.get("id"):
                operacoes.append(UpdateOne({"id": doc["id"]}, {"$set": doc}, upsert=True))
    if not operacoes:
        return 0
    resultado = db.amostras.bulk_write(operacoes, ordered=False)
    return resultado.upserted_count + resultado.modified_count


def ingerir_metricas(db, arquivos: list[Path]) -> tuple[int, int]:
    """Carrega os eventos de treino: épocas em `metricas_treino`, estado do
    gerador em `avaliacoes`. Retorna (n_epocas, n_avaliacoes)."""
    ops_m, ops_a = [], []
    for arquivo in arquivos:
        if not arquivo.exists():
            continue
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(linha)
            except json.JSONDecodeError:
                continue
            tipo = ev.get("tipo")
            if tipo == "epoca":
                chave = {"run_id": ev.get("run_id"), "epoca": ev.get("epoca")}
                ops_m.append(UpdateOne(chave, {"$set": ev}, upsert=True))
            elif tipo == "estado_gerador":
                chave = {"run_id": ev.get("run_id"), "ts": ev.get("ts")}
                ops_a.append(UpdateOne(chave, {"$set": ev}, upsert=True))
    n_m = n_a = 0
    if ops_m:
        r = db.metricas_treino.bulk_write(ops_m, ordered=False)
        n_m = r.upserted_count + r.modified_count
    if ops_a:
        r = db.avaliacoes.bulk_write(ops_a, ordered=False)
        n_a = r.upserted_count + r.modified_count
    return n_m, n_a


def resumo(db) -> None:
    """Imprime o que há hoje no banco."""
    print(f"amostras: {db.amostras.count_documents({})}"
          f" (fraude={db.amostras.count_documents({'rotulo': 'fraude'})}"
          f" | legitimo={db.amostras.count_documents({'rotulo': 'legitimo'})})")
    for campo in ("tecnica", "gerador", "fonte_dados"):
        agregado = db.amostras.aggregate([
            {"$match": {"rotulo": "fraude"}} if campo != "fonte_dados" else {"$match": {}},
            {"$group": {"_id": f"${campo}", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}])
        itens = ", ".join(f"{d['_id']}={d['n']}" for d in agregado if d["_id"])
        print(f"  por {campo}: {itens or '—'}")
    print(f"documentos distintos: {len(db.amostras.distinct('documento_id'))}")
    print(f"metricas_treino: {db.metricas_treino.count_documents({})} épocas"
          f" | runs: {len(db.metricas_treino.distinct('run_id'))}")
    print(f"avaliacoes: {db.avaliacoes.count_documents({})}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gerado", default="datasets/gerado")
    parser.add_argument("--metricas", nargs="*", default=["datasets/processed/metricas_treino.jsonl",
                                                          "datasets/processed/metricas_live.jsonl"])
    parser.add_argument("--resumo", action="store_true", help="só mostra o conteúdo do banco")
    args = parser.parse_args()

    db = conectar()
    garantir_indices(db)
    if not args.resumo:
        n = ingerir_amostras(db, Path(args.gerado))
        print(f"amostras ingeridas/atualizadas: {n}")
        n_m, n_a = ingerir_metricas(db, [Path(p) for p in args.metricas])
        print(f"métricas: {n_m} épocas | avaliações: {n_a}")
        print("-" * 50)
    resumo(db)


if __name__ == "__main__":
    main()
