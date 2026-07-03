from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from scripts.comum import deteccao
from scripts.comum.renderizacao_texto import desenhar_texto, estimar_cor_tinta, remover_texto_regiao
from scripts.geracao_fraude.manifesto import RegistroFraude

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")

PALAVRAS_CHAVE_CAMPOS = {
    "nome": ["NOME"],
    "nascimento": ["NASCIMENTO"],
    "filiacao": ["FILIACAO", "FILIAÇÃO", "MAE", "MÃE", "PAI"],
    "assinatura": ["ASSINATURA"],
}

VALORES_SUBSTITUTOS = {
    "nome": ["FULANO DA SILVA SOUZA", "BELTRANO PEREIRA LIMA", "SICRANO OLIVEIRA COSTA"],
    "nascimento": ["01/01/1990", "15/06/1985", "23/11/1978"],
    "filiacao": ["FULANA DA SILVA", "BELTRANA PEREIRA"],
    "assinatura": ["a_assinatura_e_substituida_por_rabisco"],
}


def _rabisco_assinatura(
    imagem_bgr, x: int, y: int, w: int, h: int, rng: random.Random,
    cor_bgr: tuple[int, int, int] = (20, 20, 20),
):
    resultado = imagem_bgr.copy()
    pontos = []
    n = rng.randint(6, 10)
    for i in range(n):
        px = x + int(w * i / (n - 1))
        py = y + h // 2 + rng.randint(-h // 3, h // 3)
        pontos.append((px, py))
    espessura_base = max(1, h // 12)
    for i in range(len(pontos) - 1):
        espessura = max(1, espessura_base + rng.randint(-1, 1))
        cv2.line(resultado, pontos[i], pontos[i + 1], cor_bgr, thickness=espessura, lineType=cv2.LINE_AA)
    return resultado


def processar_documento(caminho: Path, tipo_documento: str, pasta_saida: Path, rng: random.Random) -> int:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        return 0

    pasta_saida_tecnica = pasta_saida / "edicao_campos"
    pasta_saida_tecnica.mkdir(parents=True, exist_ok=True)

    total = 0
    for nome_campo, palavras_chave in PALAVRAS_CHAVE_CAMPOS.items():
        caixas = deteccao.detectar_valores_proximos_a_rotulo(imagem, palavras_chave)
        if not caixas:
            continue
        caixas_mesma_linha = [c for c in caixas if c.motivo.startswith("valor_apos_rotulo")]
        caixa = deteccao.maior_caixa(caixas_mesma_linha) or deteccao.maior_caixa(caixas)

        cor_tinta = estimar_cor_tinta(imagem, caixa.x, caixa.y, caixa.w, caixa.h)

        for fonte_correta in (True, False):
            if nome_campo == "assinatura":
                resultado = _rabisco_assinatura(
                    remover_texto_regiao(imagem, caixa.x, caixa.y, caixa.w, caixa.h), caixa.x, caixa.y, caixa.w, caixa.h, rng,
                    cor_bgr=cor_tinta,
                )
                valor_usado = "rabisco_sintetico"
            else:
                valor_usado = rng.choice(VALORES_SUBSTITUTOS[nome_campo])
                sem_texto = remover_texto_regiao(imagem, caixa.x, caixa.y, caixa.w, caixa.h)
                resultado = desenhar_texto(
                    sem_texto, valor_usado, caixa.x, caixa.y, caixa.w, caixa.h,
                    fonte_correta=fonte_correta, cor_bgr=cor_tinta,
                )

            sufixo_fonte = "fonte_correta" if fonte_correta else "fonte_incorreta"
            nome_arquivo = f"{caminho.stem}__edicao_campos_{nome_campo}_{sufixo_fonte}.jpg"
            cv2.imwrite(str(pasta_saida_tecnica / nome_arquivo), resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])

            registro = RegistroFraude(
                tecnica="edicao_campos",
                documento_origem=str(caminho),
                tipo_documento=tipo_documento,
                arquivo_gerado=nome_arquivo,
                parametros={
                    "caixa": caixa.como_tupla(),
                    "valor_substituto": valor_usado,
                    "fonte_correta": fonte_correta,
                },
                campo_alterado=nome_campo,
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
    parser.add_argument("--saida", default="datasets/fraude_gerada")
    args = parser.parse_args()

    total = processar_dataset(Path(args.legitimos), Path(args.saida))
    print(f"edicao_campos: {total} imagem(ns) gerada(s) em {Path(args.saida) / 'edicao_campos'}")


if __name__ == "__main__":
    main()
