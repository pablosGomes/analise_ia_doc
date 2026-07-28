"""Conexão com o MongoDB local e definição das coleções.

O banco roda **apenas em 127.0.0.1** (nunca exposto na rede) e guarda somente
METADADOS e MÉTRICAS — nenhuma imagem e nenhum dado pessoal. O caminho dos
documentos reais nunca é gravado: vira um hash estável (pseudonimização), de modo
que dá para agrupar/juntar amostras do mesmo documento sem registrar quem é.

Coleções:
    amostras         — uma linha por imagem gerada (técnica, gerador, rótulo, ...)
    metricas_treino  — eventos de treino por época (loss, AUC)
    avaliacoes       — fotos do estado do gerador (sonda: AUCs, contagens)
"""

from __future__ import annotations

import hashlib
import os

from pymongo import ASCENDING, MongoClient

URI_PADRAO = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
BANCO_PADRAO = os.environ.get("MONGO_DB", "iadoc")


def conectar(uri: str | None = None, banco: str | None = None, timeout_ms: int = 5000):
    """Devolve o Database do pymongo, já validando que o servidor responde."""
    cliente = MongoClient(uri or URI_PADRAO, serverSelectionTimeoutMS=timeout_ms)
    cliente.admin.command("ping")
    return cliente[banco or BANCO_PADRAO]


def id_documento(caminho_origem: str) -> str:
    """Identificador estável e NÃO reversível do documento de origem.

    Guardar o caminho real (ex.: 'datasets/legitimos/rg/WhatsApp Image ....jpeg')
    colocaria PII no banco; o hash permite agrupar todas as variantes do mesmo
    documento (necessário para split agrupado) sem registrar qual documento é.
    """
    return hashlib.sha256(str(caminho_origem).encode("utf-8")).hexdigest()[:16]


def garantir_indices(db) -> None:
    """Índices para as consultas típicas (por técnica/gerador/rótulo/documento)."""
    db.amostras.create_index([("id", ASCENDING)], unique=True)
    db.amostras.create_index([("tecnica", ASCENDING)])
    db.amostras.create_index([("gerador", ASCENDING)])
    db.amostras.create_index([("rotulo", ASCENDING)])
    db.amostras.create_index([("fonte_dados", ASCENDING)])
    db.amostras.create_index([("documento_id", ASCENDING)])
    db.metricas_treino.create_index([("run_id", ASCENDING), ("epoca", ASCENDING)])
    db.metricas_treino.create_index([("ts", ASCENDING)])
    db.avaliacoes.create_index([("ts", ASCENDING)])
