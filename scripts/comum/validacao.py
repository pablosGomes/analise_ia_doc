"""Validações de checksum de dados, compartilhadas entre `geracao_fraude/`
(para garantir que a corrupção de dígito realmente invalida o documento) e
`validacao_dados/` (camada de inferência). Mantido em paridade algorítmica
com `pablo-servico-documentos-manipulados/src/domain/validacao_dados.py`
(duplicado deliberadamente — são dois repositórios deployáveis distintos —
mas a lógica deve ser idêntica; se um mudar, replicar no outro).
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
