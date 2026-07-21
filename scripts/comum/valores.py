"""Pools de valores plausíveis (Faker pt-BR) e seleção diversificada de fontes.

Valores ou fontes fixos fariam o classificador aprender glifos/textos específicos
em vez de "campo adulterado". Por isso os valores vêm do Faker e as fontes de um
pool amplo (famílias sans/serif/mono que lembram fontes oficiais).
"""

from __future__ import annotations

import glob
import os
import random

from faker import Faker

_fake = Faker("pt_BR")


def semear(semente: int) -> None:
    """Torna o Faker reprodutível junto com o restante do pipeline."""
    Faker.seed(semente)


# --- Fontes -----------------------------------------------------------------
# Multiplataforma: procura .ttf nos diretórios de fontes do SO. Antes o caminho
# era fixo em /usr/share/fonts (só VPS/Linux), o que deixava os pools VAZIOS no
# Windows e fazia `escolher_fonte` quebrar com IndexError. Mantém a VPS igual e
# passa a funcionar no Windows da máquina local.
_DIRS_FONTES = ["/usr/share/fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
if os.name == "nt":
    _DIRS_FONTES.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))


def _todas_fontes():
    out = []
    for d in _DIRS_FONTES:
        if os.path.isdir(d):
            out += glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)
    return sorted(set(out))


_TODAS = _todas_fontes()

# Fontes REGULARES plausíveis (sem itálico/negrito), selecionadas por basename
# para não pegar variantes bold/italic por engano. Cobre Linux e Windows.
_REGULARES = {
    "arimo-regular.ttf", "liberationsans-regular.ttf", "dejavusans.ttf", "freesans.ttf",
    "tinos-regular.ttf", "liberationserif-regular.ttf", "dejavuserif.ttf", "freeserif.ttf",
    "cousine-regular.ttf", "liberationmono-regular.ttf", "dejavusansmono.ttf", "freemono.ttf",
    "arial.ttf", "tahoma.ttf", "verdana.ttf", "segoeui.ttf", "calibri.ttf",
    "times.ttf", "cour.ttf", "consola.ttf", "trebuc.ttf", "georgia.ttf",
}
# Subconjunto "família diferente porém plausível" (serif/mono) para a variante
# de incompatibilidade SUTIL — não itálico gritante.
_SUTIS = {
    "tinos-regular.ttf", "liberationserif-regular.ttf", "dejavuserif.ttf", "freeserif.ttf",
    "cousine-regular.ttf", "liberationmono-regular.ttf", "dejavusansmono.ttf", "freemono.ttf",
    "times.ttf", "georgia.ttf", "cour.ttf", "consola.ttf",
}


def _por_basename(nomes):
    return [f for f in _TODAS if os.path.basename(f).lower() in nomes]


FONTES_REGULARES = _por_basename(_REGULARES) or _TODAS
FONTES_SUTIS = _por_basename(_SUTIS) or FONTES_REGULARES


def escolher_fonte(rng, correta=True):
    pool = FONTES_REGULARES if correta else FONTES_SUTIS
    return rng.choice(pool)


# --- Valores ----------------------------------------------------------------
def nome(rng) -> str:
    return _fake.name().upper()


def filiacao(rng) -> str:
    return _fake.name().upper()


def data_nascimento(rng) -> str:
    d = _fake.date_of_birth(minimum_age=18, maximum_age=80)
    return d.strftime("%d/%m/%Y")


def _digitos_cpf(base9):
    def dv(nums):
        s = sum((len(nums) + 1 - i) * n for i, n in enumerate(nums))
        r = (s * 10) % 11
        return 0 if r == 10 else r
    d1 = dv(base9)
    d2 = dv(base9 + [d1])
    return base9 + [d1, d2]


def cpf_valido(rng) -> str:
    base = [rng.randint(0, 9) for _ in range(9)]
    nums = _digitos_cpf(base)
    return "".join(map(str, nums))
