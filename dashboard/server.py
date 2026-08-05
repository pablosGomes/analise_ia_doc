"""Servidor do dashboard de treino (roda na VPS).

Recebe as métricas do treino que roda na máquina local do Pablo (POST autenticado
por token), guarda o histórico (memória + JSONL) e serve o dashboard + um stream
SSE em tempo real. NENHUMA imagem ou PII trafega aqui — só números (loss, AUC,
contagens). A máquina local só faz requisições de saída; esta VPS nunca acessa a
máquina de casa.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
DADOS = Path(os.environ.get("DASHBOARD_DADOS", "metricas.jsonl"))
AQUI = Path(__file__).resolve().parent
# Build do front-end React (ver frontend/), gerado com `npm run build` e copiado
# para cá — `frontend/dist` -> `dist`. Não é gerado em runtime: se a pasta não
# existir, é porque o build ainda não foi feito/copiado.
DIST = AQUI / "dist"
MAX_MEM = 20000  # limite de eventos em memória

app = FastAPI(title="iadoc — dashboard de treino")
historico: list[dict] = []
clientes: set[asyncio.Queue] = set()

if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

if DADOS.exists():
    for linha in DADOS.read_text(encoding="utf-8").splitlines():
        try:
            historico.append(json.loads(linha))
        except json.JSONDecodeError:
            pass
    historico[:] = historico[-MAX_MEM:]


def _checar_token(auth: str | None) -> None:
    if TOKEN and auth != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="token inválido")


@app.post("/api/metricas")
async def receber(req: Request, authorization: str | None = Header(None)):
    _checar_token(authorization)
    evento = await req.json()
    evento["recebido_em"] = time.time()
    historico.append(evento)
    if len(historico) > MAX_MEM:
        del historico[:-MAX_MEM]
    with DADOS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    for q in list(clientes):
        try:
            q.put_nowait(evento)
        except asyncio.QueueFull:
            pass
    return {"ok": True}


@app.get("/api/estado")
async def estado():
    return JSONResponse(historico[-MAX_MEM:])


@app.get("/api/stream")
async def stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    clientes.add(q)

    async def gerar():
        try:
            yield f"event: snapshot\ndata: {json.dumps(historico[-MAX_MEM:])}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # heartbeat (mantém a conexão viva por proxies)
        finally:
            clientes.discard(q)

    return StreamingResponse(gerar(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/saude")
async def saude():
    return {"ok": True, "eventos": len(historico), "clientes": len(clientes)}


@app.get("/")
async def index():
    return HTMLResponse((DIST / "index.html").read_text(encoding="utf-8"))
