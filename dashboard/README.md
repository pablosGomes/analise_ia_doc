# Dashboard de treino (iadoc)

Retorno visual em tempo real do treino, com o **backend (treino) rodando na máquina
local** (GPU) e o **frontend hospedado na VPS**. A máquina local só faz requisições
de saída — a VPS nunca acessa a máquina de casa. **Nenhuma imagem ou PII trafega:
só números** (loss, AUC, contagens).

```
[máquina local / GPU]                         [VPS: iadoc.taberna3d.com.br]
  treinar_ao_vivo  --(HTTPS POST, token)-->     server.py (FastAPI, :8090)
   └ Publicador                                   ├ guarda histórico (metricas.jsonl)
                                                  ├ serve dist/ (React) em /
                                                  └ stream SSE em /api/stream (tempo real)
```

O front-end é uma SPA em **React + TypeScript** (Vite), buildada localmente e
implantada como arquivos estáticos — a VPS não precisa de Node/npm instalado.

## Componentes
- `../scripts/treino/treinar_ao_vivo.py` — treino por épocas (MLP sobre embeddings DinoV2), emite métricas.
- `../scripts/treino/publicador.py` — grava JSONL local e faz POST best-effort à VPS.
- `server.py` — FastAPI: recebe POSTs autenticados, serve o build do front (`dist/`) + stream SSE.
- `frontend/` — código-fonte do painel (React + TypeScript + Vite + recharts). Versionado; `frontend/dist/` e `frontend/node_modules/` não são (`.gitignore`).
- `dist/` — build do front pronto para servir (`frontend/dist/` copiado para cá). Não versionado — refeito a cada deploy.

### Estrutura do front-end
```
frontend/src/
  types.ts              tipos dos eventos do stream (espelham o payload do backend)
  lib/                   formato.ts (fmt/faixa), nomes.ts (tradução de chaves), paleta.ts (cores dos gráficos)
  hooks/useTreinoStream.ts  conecta ao /api/stream (SSE) e mantém o estado ao vivo
  components/            Header, Hero (veredito), ComoLerNotas, TestesGeneralizacao,
                          BaseTreino, Graficos (recharts), Card/Row/Gauge/Note (reutilizáveis)
```

## Desenvolvimento local

Duas partes rodando ao mesmo tempo:

```bash
# 1) o backend (a partir de dashboard/)
uvicorn server:app --reload --port 8090

# 2) o front, com hot-reload (a partir de dashboard/frontend/)
npm install   # primeira vez
npm run dev   # abre em http://localhost:5173, com /api em proxy para :8090
```

`npm run typecheck` roda só a checagem de tipos, sem build.

## Build e deploy

```bash
# a partir de dashboard/frontend/
npm run build                                   # gera frontend/dist/
rm -rf ../dist && cp -r dist ../dist             # copia para onde o server.py espera

# envia para a VPS e reinicia o serviço
scp -r ../dist ../server.py root@vps:/root/apps/iadoc-dashboard/
ssh root@vps systemctl restart iadoc-dashboard
```

- App em `/root/apps/iadoc-dashboard/` na porta **8090** (loopback).
- systemd: `iadoc-dashboard.service` (token em `.env`, chmod 600).
- nginx: site `iadoc.taberna3d.com.br` faz proxy → 8090, com `proxy_buffering off` + `proxy_read_timeout 3600s` para o SSE. HTTPS via Let's Encrypt; fronteado por Cloudflare.

## Rodar o treino publicando no dashboard (máquina local)
```bash
export DASHBOARD_URL=https://iadoc.taberna3d.com.br
export DASHBOARD_TOKEN=<token>      # está em /root/apps/iadoc-dashboard/.env na VPS
python -m scripts.treino.treinar_ao_vivo --epocas 120 --intervalo 0.3
```
Sem `DASHBOARD_URL`, o treino roda igual e só grava o JSONL local.

## Notas
- O `Publicador` envia `User-Agent: iadoc-trainer/1.0` — o padrão do urllib é barrado (403) pelo Cloudflare.
- A visualização é pública (só métricas, sem PII). Para restringir, adicionar HTTP Basic Auth no nginx.
- Mudar o formato de um evento em `treinar_ao_vivo.py`/`publicador.py`? Atualizar `frontend/src/types.ts` junto — os tipos são um espelho manual do payload real, não gerados.
