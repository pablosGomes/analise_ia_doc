import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.comum.validacao import validar_cpf
from scripts.geracao_fraude.digito_verificador import _corromper_digitos


def test_corromper_cpf_sempre_produz_checksum_invalido():
    cpf_valido = "11144477735"
    rng = random.Random(0)
    for _ in range(30):
        corrompido = _corromper_digitos(cpf_valido, rng)
        assert len(corrompido) == 11
        assert validar_cpf(corrompido) is False


def test_corromper_cpf_altera_apenas_um_digito_entre_os_9_primeiros():
    cpf_valido = "11144477735"
    rng = random.Random(1)
    corrompido = _corromper_digitos(cpf_valido, rng)
    diferencas = [i for i in range(11) if cpf_valido[i] != corrompido[i]]
    assert len(diferencas) == 1
    assert diferencas[0] < 9  # nunca altera diretamente um dos 2 dígitos verificadores


def test_corromper_numero_generico_preserva_tamanho():
    numero = "1234567890123"  # 13 dígitos, não é CPF
    rng = random.Random(2)
    corrompido = _corromper_digitos(numero, rng)
    assert len(corrompido) == len(numero)
    assert corrompido != numero
