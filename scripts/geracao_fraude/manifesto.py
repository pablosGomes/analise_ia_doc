"""Esquema de manifesto para exemplos gerados a partir de documentos legítimos.

Cada saída (fraude ou legítimo processado) grava um JSON ao lado da imagem, com a
origem, a técnica e parâmetros, o rótulo e os parâmetros do pipeline de captura.
Necessário para (a) auditoria e reprodutibilidade, (b) validação cruzada por
técnica e (c) atender a pedidos de revogação de consentimento.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class RegistroFraude:
    tecnica: str
    documento_origem: str
    tipo_documento: str
    arquivo_gerado: str
    parametros: dict[str, Any] = field(default_factory=dict)
    campo_alterado: Optional[str] = None
    dificuldade: str = "media"
    # --- rótulo e metadados de captura/manipulação ---
    rotulo: str = "fraude"                     # "fraude" | "legitimo"
    parametros_captura: dict[str, Any] = field(default_factory=dict)
    metodo_inpaint: Optional[str] = None
    fonte: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def caminho_manifesto(self, diretorio_saida: Path) -> Path:
        return diretorio_saida / f"{Path(self.arquivo_gerado).stem}.manifesto.json"

    def salvar(self, diretorio_saida: Path) -> Path:
        diretorio_saida.mkdir(parents=True, exist_ok=True)
        caminho = self.caminho_manifesto(diretorio_saida)
        caminho.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return caminho


def carregar(caminho_manifesto: Path) -> RegistroFraude:
    dados = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    # ignora chaves desconhecidas (tolera manifestos de versões anteriores do esquema)
    campos_validos = RegistroFraude.__dataclass_fields__.keys()
    dados = {k: v for k, v in dados.items() if k in campos_validos}
    return RegistroFraude(**dados)


def listar_manifestos(diretorio: Path) -> list[RegistroFraude]:
    return [carregar(p) for p in sorted(diretorio.glob("*.manifesto.json"))]
