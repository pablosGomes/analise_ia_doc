"""Garante que a ingestão no MongoDB não grava PII.

Estes testes não precisam de banco: exercitam a montagem do documento a partir de
um manifesto, que é onde a sanitização acontece.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.db.ingestao import _amostra_do_manifesto, _sanitizar_arquivo, _sanitizar_valor
from scripts.db.mongo import id_documento

MANIFESTO_REAL = {
    "id": "abc-123",
    "tecnica": "troca_foto",
    "documento_origem": r"datasets\legitimos\rg\WhatsApp Image 2026-07-02 at 11.13.21.jpeg",
    "tipo_documento": "rg",
    "arquivo_gerado": "WhatsApp Image 2026-07-02 at 11.13.21__troca_foto_alinhado_0.png",
    "rotulo": "fraude",
    "gerador": "landmark_classico",
    "parametros": {"modo": "alinhado",
                   "fonte_rosto": r"datasets\legitimos\cnh\WhatsApp Image 2026-07-02 at 11.13.00.jpeg",
                   "texto_original": "NOME REAL DA PESSOA"},
}


def _escrever(tmp_path, manifesto):
    p = tmp_path / "x.manifesto.json"
    p.write_text(json.dumps(manifesto), encoding="utf-8")
    return p


def test_documento_origem_nunca_e_gravado(tmp_path):
    doc = _amostra_do_manifesto(_escrever(tmp_path, MANIFESTO_REAL), "reais")
    texto = json.dumps(doc)
    assert "WhatsApp" not in texto
    assert "legitimos" not in texto
    assert ".jpeg" not in texto
    assert doc["documento_id"] == id_documento(MANIFESTO_REAL["documento_origem"])


def test_caminho_em_parametros_vira_hash(tmp_path):
    doc = _amostra_do_manifesto(_escrever(tmp_path, MANIFESTO_REAL), "reais")
    assert doc["parametros"]["fonte_rosto"] == id_documento(MANIFESTO_REAL["parametros"]["fonte_rosto"])
    assert doc["parametros"]["modo"] == "alinhado"          # campo normal preservado


def test_texto_original_descartado_em_reais(tmp_path):
    doc = _amostra_do_manifesto(_escrever(tmp_path, MANIFESTO_REAL), "reais")
    assert "texto_original" not in doc["parametros"]


def test_texto_original_mantido_no_bid(tmp_path):
    m = dict(MANIFESTO_REAL, documento_origem="datasets/BID Dataset/RG_Aberto/00000001_in.jpg",
             arquivo_gerado="00000001__rg__edicao_0_nome.png",
             parametros={"texto_original": "NOME SINTETICO", "origem": "bid"})
    doc = _amostra_do_manifesto(_escrever(tmp_path, m), "bid")
    assert doc["parametros"]["texto_original"] == "NOME SINTETICO"   # BID é sintético


def test_sanitizar_arquivo_troca_prefixo():
    assert _sanitizar_arquivo("WhatsApp Image 2026 at 11.13__edicao_campos_nome.png", "deadbeef") \
        == "deadbeef__edicao_campos_nome.png"


def test_sanitizar_valor_preserva_nao_caminhos():
    assert _sanitizar_valor({"caixa": [1, 2, 3], "valor": "MARIA SILVA", "conf": 0.9}) \
        == {"caixa": [1, 2, 3], "valor": "MARIA SILVA", "conf": 0.9}
