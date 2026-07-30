"""Estatísticas de inconsistência local sobre a grade de patches do DinoV2.

O vetor global (token CLS) resume o documento inteiro num único ponto, e uma
adulteração que afeta dois ou três patches é diluída até desaparecer — foi o que a
medição mostrou: a edição de campo fica no acaso mesmo com o documento inteiro em
alta resolução.

Aqui a grade de patches é usada diretamente. Para cada patch mede-se o quanto ele
destoa (a) dos oito vizinhos imediatos e (b) do documento como um todo. A adulteração
é LOCAL: ela produz um pico nesse mapa, não uma mudança na média. Por isso as
estatísticas agregadas privilegiam os extremos (máximo, percentis altos, razão
pico/média) em vez da média.

Saída: um vetor compacto por imagem, que pode ser usado sozinho ou concatenado ao
embedding global.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Nomes das estatísticas, na ordem em que `estatisticas_anomalia` as devolve.
NOMES = (
    "local_max", "local_p99", "local_p95", "local_media", "local_desvio", "local_pico_media",
    "global_max", "global_p99", "global_p95", "global_media", "global_desvio", "global_pico_media",
)


def _mapa_desvio_local(grade: torch.Tensor) -> torch.Tensor:
    """Distância de cosseno entre cada patch e a média dos 8 vizinhos.

    `grade`: (B, D, G, G) já normalizada em D. Retorna (B, G, G).
    A borda usa padding por replicação — com zeros, os patches da borda teriam
    vizinhança artificialmente diferente e virariam falsos picos.
    """
    com_borda = F.pad(grade, (1, 1, 1, 1), mode="replicate")
    soma_9 = F.avg_pool2d(com_borda, kernel_size=3, stride=1) * 9.0
    vizinhos = F.normalize((soma_9 - grade) / 8.0, dim=1)
    return 1.0 - (grade * vizinhos).sum(dim=1)


def _mapa_desvio_global(grade: torch.Tensor) -> torch.Tensor:
    """Distância de cosseno entre cada patch e o patch médio do documento."""
    media = F.normalize(grade.mean(dim=(2, 3)), dim=1)          # (B, D)
    return 1.0 - torch.einsum("bdij,bd->bij", grade, media)


def _resumir(mapa: torch.Tensor) -> torch.Tensor:
    """Resume um mapa (B, G, G) em 6 estatísticas por imagem.

    Privilegia extremos: uma edição local eleva o topo da distribuição, quase não
    move a média. `pico_media` (máximo ÷ média) mede o quanto a região mais
    destoante se destaca do resto do documento.
    """
    plano = mapa.flatten(1)                                     # (B, G*G)
    maximo = plano.max(dim=1).values
    p99 = plano.quantile(0.99, dim=1)
    p95 = plano.quantile(0.95, dim=1)
    media = plano.mean(dim=1)
    desvio = plano.std(dim=1)
    pico_media = maximo / (media + 1e-6)
    return torch.stack([maximo, p99, p95, media, desvio, pico_media], dim=1)


@torch.inference_mode()
def estatisticas_anomalia(patches: torch.Tensor) -> torch.Tensor:
    """(B, N, D) de tokens de patch -> (B, 12) de estatísticas de inconsistência.

    `patches` não deve incluir o token CLS. N precisa ser um quadrado perfeito.
    """
    B, N, D = patches.shape
    lado = int(round(N ** 0.5))
    if lado * lado != N:
        raise ValueError(f"grade de patches não é quadrada: N={N}")
    grade = F.normalize(patches, dim=-1).transpose(1, 2).reshape(B, D, lado, lado)
    return torch.cat([_resumir(_mapa_desvio_local(grade)),
                      _resumir(_mapa_desvio_global(grade))], dim=1)
