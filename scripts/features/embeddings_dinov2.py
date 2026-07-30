"""Extração de embeddings via DinoV2 pré-treinado (Fase 3 / Fase A do roadmap).

Percorre as imagens geradas em ``datasets/gerado/`` (fraude e legítimos, do BID e
dos documentos reais), extrai o embedding de cada uma com o backbone DinoV2
pré-treinado (``facebook/dinov2-base``, 768 dimensões) e grava um único arquivo
``.npz`` com a matriz de embeddings mais uma tabela de metadados alinhada linha a
linha.

Os metadados incluem tudo que o treino precisa para montar splits corretos e a
validação anti-atalho: rótulo (0 legítimo / 1 fraude), técnica, documento de
origem (para split agrupado, evitando vazamento entre variantes do mesmo
documento) e a fonte do dado (bid / reais).

Uso:
    python -m scripts.features.embeddings_dinov2 \
        --entrada datasets/gerado --saida datasets/processed/embeddings_dinov2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from scripts.features.anomalia import NOMES as NOMES_ANOMALIA, estatisticas_anomalia

MODELO_PADRAO = "facebook/dinov2-base"

# Resolução de entrada. O pré-processador padrão do DinoV2 reduz o menor lado para
# 256 e faz um CENTER CROP de 224x224 — o que descarta ~40% da área do documento e
# deixa um campo de texto de ~12 px com menos de 3 px de altura, tornando a edição
# de texto fisicamente indetectável. Aqui a imagem inteira é redimensionada para um
# quadrado de lado `RESOLUCAO_PADRAO`, sem corte. 518 é o valor de alta resolução
# usado pelo próprio DinoV2 em tarefas densas (múltiplo do patch de 14 -> 37x37).
RESOLUCAO_PADRAO = 518

# Subpastas de datasets/gerado e a fonte (bid/reais) que cada uma representa.
GRUPOS = {
    "bid_fraude": "bid",
    "bid_legitimos": "bid",
    "reais_fraude": "reais",
    "reais_legitimos": "reais",
}


def _carregar_manifesto(caminho_imagem: Path) -> dict:
    """Lê o manifesto irmão da imagem, se existir. Retorna {} quando ausente."""
    manifesto = caminho_imagem.with_name(f"{caminho_imagem.stem}.manifesto.json")
    if not manifesto.exists():
        return {}
    try:
        return json.loads(manifesto.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _indexar(pasta_entrada: Path) -> list[dict]:
    """Monta a lista de amostras (imagem + metadados) a partir das subpastas."""
    amostras: list[dict] = []
    for subpasta, fonte in GRUPOS.items():
        raiz = pasta_entrada / subpasta
        if not raiz.exists():
            continue
        for caminho in sorted(raiz.rglob("*.png")):
            manifesto = _carregar_manifesto(caminho)
            rotulo_texto = manifesto.get("rotulo")
            if rotulo_texto is None:
                # Sem manifesto: inferir rótulo pela subpasta (…_fraude / …_legitimos).
                rotulo_texto = "fraude" if "fraude" in subpasta else "legitimo"
            amostras.append({
                "caminho": str(caminho),
                "rotulo": 1 if rotulo_texto == "fraude" else 0,
                "tecnica": manifesto.get("tecnica", caminho.parent.name),
                "documento_origem": manifesto.get("documento_origem", ""),
                "tipo_documento": manifesto.get("tipo_documento", ""),
                "gerador": manifesto.get("gerador") or "",
                "fonte": fonte,
            })
    return amostras


def _carregar_modelo(nome_modelo: str, dispositivo: str):
    processador = AutoImageProcessor.from_pretrained(nome_modelo)
    modelo = AutoModel.from_pretrained(nome_modelo).to(dispositivo).eval()
    return processador, modelo


@torch.inference_mode()
def _extrair_lote(caminhos: list[str], processador, modelo, dispositivo: str,
                  resolucao: int) -> np.ndarray:
    # Redimensiona o documento INTEIRO para o quadrado de entrada e desliga o resize
    # e o center crop do processador (ver nota em RESOLUCAO_PADRAO). A distorção de
    # proporção é idêntica para fraude e legítimo do mesmo documento, então não cria
    # atalho; o corte, ao descartar regiões, criaria.
    imagens = [Image.open(c).convert("RGB").resize((resolucao, resolucao), Image.BICUBIC)
               for c in caminhos]
    entradas = processador(images=imagens, do_resize=False, do_center_crop=False,
                           return_tensors="pt").to(dispositivo)
    saidas = modelo(**entradas)
    # pooler_output = token CLS após layernorm; representação global da imagem.
    cls = saidas.pooler_output.detach().float()
    # Estatísticas de inconsistência local sobre a grade de patches (descarta o CLS,
    # que é o primeiro token): capturam adulteração LOCAL, que o vetor global dilui.
    anomalia = estatisticas_anomalia(saidas.last_hidden_state[:, 1:].detach().float())
    return cls.cpu().numpy(), anomalia.cpu().numpy()


def extrair(pasta_entrada: Path, arquivo_saida: Path, nome_modelo: str,
            tamanho_lote: int, resolucao: int = RESOLUCAO_PADRAO) -> None:
    amostras = _indexar(pasta_entrada)
    if not amostras:
        raise SystemExit(f"Nenhuma imagem .png encontrada em {pasta_entrada}")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    n_fraude = sum(a["rotulo"] for a in amostras)
    print(f"Amostras: {len(amostras)} (fraude={n_fraude} | legitimo={len(amostras) - n_fraude})")
    print(f"Modelo: {nome_modelo} | dispositivo: {dispositivo} | lote: {tamanho_lote}"
          f" | resolucao: {resolucao}x{resolucao} (documento inteiro, sem crop)")

    processador, modelo = _carregar_modelo(nome_modelo, dispositivo)

    embeddings: list[np.ndarray] = []
    anomalias: list[np.ndarray] = []
    total = len(amostras)
    for inicio in range(0, total, tamanho_lote):
        lote = amostras[inicio:inicio + tamanho_lote]
        emb_lote, ano_lote = _extrair_lote([a["caminho"] for a in lote], processador, modelo,
                                           dispositivo, resolucao)
        embeddings.append(emb_lote); anomalias.append(ano_lote)
        processadas = min(inicio + tamanho_lote, total)
        print(f"\r  {processadas}/{total} imagens", end="", flush=True)
    print()

    X = np.concatenate(embeddings, axis=0).astype(np.float32)
    X_anomalia = np.concatenate(anomalias, axis=0).astype(np.float32)
    y = np.array([a["rotulo"] for a in amostras], dtype=np.int64)
    tecnica = np.array([a["tecnica"] for a in amostras])
    documento_origem = np.array([a["documento_origem"] for a in amostras])
    tipo_documento = np.array([a["tipo_documento"] for a in amostras])
    gerador = np.array([a["gerador"] for a in amostras])
    fonte = np.array([a["fonte"] for a in amostras])
    caminho = np.array([a["caminho"] for a in amostras])

    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arquivo_saida,
        X=X, X_anomalia=X_anomalia, nomes_anomalia=np.array(NOMES_ANOMALIA), y=y, tecnica=tecnica, documento_origem=documento_origem,
        tipo_documento=tipo_documento, gerador=gerador, fonte=fonte, caminho=caminho,
        modelo=nome_modelo, resolucao=resolucao,
    )
    print(f"Embeddings salvos: {arquivo_saida}  (X={X.shape}, anomalia={X_anomalia.shape})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", default="datasets/gerado")
    parser.add_argument("--saida", default="datasets/processed/embeddings_dinov2.npz")
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--tamanho-lote", type=int, default=16)
    parser.add_argument("--resolucao", type=int, default=RESOLUCAO_PADRAO,
                        help="lado do quadrado de entrada; multiplo de 14 (patch do ViT)")
    args = parser.parse_args()
    extrair(Path(args.entrada), Path(args.saida), args.modelo, args.tamanho_lote, args.resolucao)


if __name__ == "__main__":
    main()
