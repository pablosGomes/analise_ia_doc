"""Finalização compartilhada de toda amostra gerada (fraude ou legítimo).

Aplica o simulador de captura compartilhado (paridade de pipeline) e salva em PNG
(contêiner sem perda), garantindo que o único artefato de compressão presente seja
o JPEG aleatório de dentro da captura, idêntico em distribuição para as duas classes.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from scripts.comum import captura


def finalizar(imagem_limpa, rng, pasta_saida: Path, nome_stem: str, registro):
    """Aplica a captura, salva PNG + manifesto e retorna o caminho da imagem."""
    imagem_cap, params_cap = captura.simular_captura(imagem_limpa, rng)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    nome_arquivo = f"{nome_stem}.png"
    cv2.imwrite(str(pasta_saida / nome_arquivo), imagem_cap)
    registro.arquivo_gerado = nome_arquivo
    registro.parametros_captura = params_cap
    registro.salvar(pasta_saida)
    return pasta_saida / nome_arquivo
