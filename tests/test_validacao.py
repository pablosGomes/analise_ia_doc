import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.comum.validacao import limpar_digitos, validar_cpf


def test_limpar_digitos_remove_pontuacao():
    assert limpar_digitos("123.456.789-09") == "12345678909"


def test_validar_cpf_aceita_cpf_valido_conhecido():
    # CPF de teste amplamente usado em exemplos públicos de validação de checksum
    assert validar_cpf("111.444.777-35") is True


def test_validar_cpf_rejeita_todos_digitos_iguais():
    assert validar_cpf("111.111.111-11") is False


def test_validar_cpf_rejeita_tamanho_incorreto():
    assert validar_cpf("123456") is False


def test_validar_cpf_rejeita_digito_verificador_alterado():
    cpf_valido = "11144477735"
    ultimo_digito_alterado = cpf_valido[:-1] + str((int(cpf_valido[-1]) + 1) % 10)
    assert validar_cpf(ultimo_digito_alterado) is False
