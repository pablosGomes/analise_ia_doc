import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.geracao_fraude.manifesto import RegistroFraude, carregar, listar_manifestos


def test_salvar_e_carregar_roundtrip(tmp_path):
    registro = RegistroFraude(
        tecnica="troca_foto",
        documento_origem="datasets/legitimos/rg/exemplo.jpg",
        tipo_documento="rg",
        arquivo_gerado="exemplo__troca_foto_alinhado_0.png",
        parametros={"modo": "alinhado", "confianca_alvo": 0.9, "confianca_fonte": 0.8},
        campo_alterado="foto_rosto",
        dificuldade="sutil",
        rotulo="fraude",
    )
    caminho = registro.salvar(tmp_path)
    assert caminho.exists()

    recarregado = carregar(caminho)
    assert recarregado.tecnica == "troca_foto"
    assert recarregado.arquivo_gerado == "exemplo__troca_foto_alinhado_0.png"
    assert recarregado.parametros["modo"] == "alinhado"
    assert recarregado.id == registro.id


def test_listar_manifestos_retorna_todos(tmp_path):
    for i in range(3):
        RegistroFraude(
            tecnica="edicao_campos",
            documento_origem="doc.jpg",
            tipo_documento="cnh",
            arquivo_gerado=f"doc__edicao_campos_{i}.png",
        ).salvar(tmp_path)

    manifestos = listar_manifestos(tmp_path)
    assert len(manifestos) == 3
    assert all(m.tecnica == "edicao_campos" for m in manifestos)


def test_ids_gerados_sao_unicos():
    registros = [RegistroFraude(tecnica="t", documento_origem="d", tipo_documento="rg", arquivo_gerado="a.jpg") for _ in range(50)]
    ids = {r.id for r in registros}
    assert len(ids) == 50
