"""Técnica 3 — Corrupção de dígito verificador.

Localiza um número de documento (CPF de 11 dígitos, ou outro número de
registro) via OCR, altera um único dígito do número original — mantendo o
restante idêntico — e redesenha o campo. Quando o número é um CPF, o dígito
alterado é escolhido entre os 9 primeiros (nunca os 2 dígitos verificadores
diretamente), simulando o cenário mais realista de fraude: alguém edita o
número mas não recalcula o checksum. `scripts/comum/validacao.validar_cpf`
é usada para garantir que o resultado É de fato inválido (repete a escolha
de dígito no raro caso de coincidência).

Esta técnica treina especificamente a camada de imagem a reconhecer a
*edição* de um campo numérico (textura/inpaint/fonte), independente de saber
o valor do checksum — a validação do checksum em si é responsabilidade da
camada de dados (`validacao_dados/`), que roda sobre o texto extraído por
OCR, não sobre a imagem.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from scripts.comum import deteccao
from scripts.comum.renderizacao_texto import desenhar_texto, estimar_cor_tinta, remover_texto_regiao
from scripts.comum.validacao import limpar_digitos, validar_cpf
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def _corromper_digitos(digitos: str, rng: random.Random) -> str:
    """Altera um único dígito. Se o resultado for um CPF (11 dígitos) e por
    coincidência ainda for válido, tenta novamente com outra posição/valor."""
    eh_cpf = len(digitos) == 11
    posicoes_candidatas = list(range(9)) if eh_cpf else list(range(len(digitos)))

    for _ in range(20):
        pos = rng.choice(posicoes_candidatas)
        original = digitos[pos]
        novo_digito = rng.choice([d for d in "0123456789" if d != original])
        corrompido = digitos[:pos] + novo_digito + digitos[pos + 1:]
        if not eh_cpf or not validar_cpf(corrompido):
            return corrompido
    return corrompido  # extremamente improvável de chegar aqui


def processar_documento(caminho: Path, tipo_documento: str, pasta_saida: Path, rng: random.Random) -> int:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0

    caixas_numero = deteccao.detectar_numeros_documento(imagem)
    if not caixas_numero:
        return 0

    pasta_saida_tecnica = pasta_saida / "digito_verificador"
    pasta_saida_tecnica.mkdir(parents=True, exist_ok=True)

    total = 0
    for idx, caixa in enumerate(caixas_numero):
        digitos_originais = limpar_digitos(caixa.texto)
        if len(digitos_originais) < 5:
            continue  # sequência curta demais para ser um número de documento confiável

        # Amostra a cor de tinta real ANTES do inpaint apagar o texto original
        # — mais fiel que preto fixo, e consistente entre as duas variantes.
        cor_tinta = estimar_cor_tinta(imagem, caixa.x, caixa.y, caixa.w, caixa.h)

        for fonte_correta in (True, False):
            digitos_corrompidos = _corromper_digitos(digitos_originais, rng)

            sem_texto = remover_texto_regiao(imagem, caixa.x, caixa.y, caixa.w, caixa.h)
            resultado = desenhar_texto(
                sem_texto, digitos_corrompidos, caixa.x, caixa.y, caixa.w, caixa.h,
                fonte_correta=fonte_correta, cor_bgr=cor_tinta,
            )

            sufixo_fonte = "fonte_correta" if fonte_correta else "fonte_incorreta"
            nome_arquivo = f"{caminho.stem}__digito_verificador_{idx}_{sufixo_fonte}.jpg"
            cv2.imwrite(str(pasta_saida_tecnica / nome_arquivo), resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])

            registro = RegistroFraude(
                tecnica="digito_verificador",
                documento_origem=str(caminho),
                tipo_documento=tipo_documento,
                arquivo_gerado=nome_arquivo,
                parametros={
                    "caixa": caixa.como_tupla(),
                    "quantidade_digitos": len(digitos_originais),
                    "fonte_correta": fonte_correta,
                    "eh_cpf_11_digitos": len(digitos_originais) == 11,
                },
                campo_alterado="numero_documento",
                dificuldade="sutil" if fonte_correta else "evidente",
            )
            registro.salvar(pasta_saida_tecnica)
            total += 1
    return total


def processar_dataset(pasta_legitimos: Path, pasta_saida: Path, semente: int = 42) -> int:
    rng = random.Random(semente)
    total = 0
    for pasta_tipo in sorted(pasta_legitimos.iterdir()):
        if not pasta_tipo.is_dir():
            continue
        for caminho in sorted(p for p in pasta_tipo.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM):
            total += processar_documento(caminho, pasta_tipo.name, pasta_saida, rng)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legitimos", default="datasets/legitimos")
    parser.add_argument