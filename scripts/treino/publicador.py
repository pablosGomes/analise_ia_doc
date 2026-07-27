"""Publicador de métricas de treino.

Grava cada evento (início, época, fim, estado do gerador) num JSONL local e,
se um dashboard estiver configurado (via env `DASHBOARD_URL` + `DASHBOARD_TOKEN`,
ou parâmetros), também envia por HTTPS POST autenticado. O envio é sempre
"best-effort": qualquer falha de rede é engolida e logada — NUNCA interrompe o
treino. Assim o treino roda igual na máquina local com ou sem a VPS no ar.

Fluxo desejado: máquina local (GPU) treina → publica (outbound) → servidor na VPS
recebe e serve o dashboard. A máquina local nunca precisa aceitar conexões de fora.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path


class Publicador:
    def __init__(self, caminho_jsonl, url=None, token=None, run_id=None, timeout=5):
        self.caminho = Path(caminho_jsonl)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.url = (url or os.environ.get("DASHBOARD_URL") or "").rstrip("/") or None
        self.token = token or os.environ.get("DASHBOARD_TOKEN")
        self.run_id = run_id or f"run-{int(time.time())}"
        self.timeout = timeout
        self._avisou_falha = False

    def publicar(self, evento: dict) -> None:
        """Anexa o evento ao JSONL local e tenta enviar ao dashboard (best-effort)."""
        evento = {"run_id": self.run_id, "ts": time.time(), **evento}
        with self.caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        if not self.url:
            return
        try:
            dados = json.dumps(evento).encode("utf-8")
            # User-Agent explícito: o padrão do urllib ("Python-urllib/...") é
            # barrado como bot pelo Cloudflare (403) na frente da VPS.
            cabecalhos = {"Content-Type": "application/json", "User-Agent": "iadoc-trainer/1.0"}
            if self.token:
                cabecalhos["Authorization"] = f"Bearer {self.token}"
            req = urllib.request.Request(self.url + "/api/metricas", data=dados,
                                         headers=cabecalhos, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception as e:
            if not self._avisou_falha:  # avisa uma vez, não polui o log de treino
                print(f"[publicador] dashboard inacessível, seguindo só com JSONL local: {e}")
                self._avisou_falha = True
