"""Validações de checksum de dados (camada de dados, independente da imagem).

`validar_cpf` confere os dígitos verificadores do CPF — a checagem que pega uma
fraude mesmo com a imagem perfeita. É a base da camada de validação de dados
(`scripts/validacao_dados/`, Fase 4 do roadmap), separada do sinal de imagem.
"""

from __future__ import annotations


def limpar_digitos(texto: str) -> str:
    return "".join(c for c in texto if c.isdigit())


def validar_cpf(cpf: str) -> bool:
    """Valida os dígitos verificadores do CPF (algoritmo público de checksum)."""
    digitos = [int(c) for c in limpar_digitos(cpf)]
    if len(digitos) != 11 or len(set(digitos)) == 1:
        return False

    def _digito_verificador(nums: list[int], peso_inicial: int) -> int:
        soma = sum(n * p for n, p in zip(nums, range(peso_inicial, 1, -1)))
        resto = (soma * 10) % 11
        return 0 if resto == 10 else resto

    d1 = _digito_verificador(digitos[:9], 10)
    d2 = _digito_verificador(digitos[:9] + [d1], 11)
    return digitos[-2:] == [d1, d2]
