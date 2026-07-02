import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.geracao_fraude.manifesto import RegistroFraude, carregar, listar_manifestos


def test_salvar_e_carregar_roundtrip(tmp_path):
    registro = RegistroFraude(
        tecnica="crop_and_move",
        documento_origem="datasets/legitimos/rg/exemplo.jpg",
        tipo_documento="rg",
        arquivo_gerado="exemplo__crop_and_move_duplicacao_0.jpg",
        parametros={"modo": "duplicacao", "origem": [1, 2, 3, 4], "destino": [5, 6, 7, 8]},
        campo_alterado="regiao_arbitraria",
        dificuldade="media",
    )
    caminho = registro.salvar(tmp_path)
    assert caminho.exists()

    recarregado = carregar(caminho)
    assert recarregado.tecnica == "crop_and_move"
    assert recarregado.arquivo_gerado == "exemplo__crop_and_move_duplicacao_0.jpg"
    assert recarregado.parametros["modo"] == "duplicacao"
    assert recarregado.id == registro.id


def test_listar_manifestos_retorna_todos(tmp_path):
    for i in range(3):
        RegistroFraude(
            tecnica="digito_verificador",
            documento_origem="doc.jpg",
            tipo_documento="cnh",
            arquivo_gerado=f"doc__digito_verificador_{i}.jpg",
        ).salvar(tmp_path)

    manifestos = listar_manifestos(tmp_path)
    assert len(manifestos) == 3
    assert all(m.tecnica == "digito_verificador" for m in manifestos)


def test_ids_gerados_sao_unicos():
    registros = [RegistroFraude(tecnica="t", documento_origem="d", tipo_documento="rg", arquivo_gerado="a.jpg") for _ in range(50)]
    ids = {r.id for r in registros}
    assert len(ids) == 50
