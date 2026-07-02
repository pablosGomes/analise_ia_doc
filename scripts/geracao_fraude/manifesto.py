"""Esquema de manifesto para exemplos de fraude gerados a partir de documentos
legítimos. Cada variante de fraude grava um registro JSON ao lado da imagem
gerada, permitindo rastrear de qual documento legítimo ela se origina, qual
técnica/parâmetros foram aplicados e qual campo foi alterado — necessário
para (a) auditoria/reprodutibilidade, (b) a etapa de treino poder separar
por técnica na validação cruzada (mitigar o "Synthetic Utility Gap", ver
docs/documentacao_deteccao_fraude.docx Seção 7.2), e (c) atender pedidos de
revogação de consentimento (remover todas as variantes derivadas de um
documento de um titular específico).
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
    documento_origem: str          # caminho relativo (a partir de datasets/) do legítimo de origem
    tipo_documento: str            # rg | cnh | passaporte
    arquivo_gerado: str            # nome do arquivo de imagem gerado (relativo à pasta da técnica)
    parametros: dict[str, Any] = field(default_factory=dict)
    campo_alterado: Optional[str] = None
    dificuldade: str = "media"     # "sutil" | "media" | "evidente" — dificuldade esperada de detecção
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
    return RegistroFraude(**dados)


def listar_manifestos(diretorio: Path) -> list[RegistroFraude]:
    return [carregar(p) for p in sorted(diretorio.glob("*.manifesto.json"))]
