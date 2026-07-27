# Dashboard de treino (iadoc)

Retorno visual em tempo real do treino, com o **backend (treino) rodando na máquina
local** (GPU) e o **frontend hospedado na VPS**. A máquina local só faz requisições
de saída — a VPS nunca acessa a máquina de casa. **Nenhuma imagem ou PII trafega:
só números** (loss, AUC, contagens).

```
[máquina local / GPU]                         [VPS: iadoc.taberna3d.com.br]
  treinar_ao_vivo  --(HTTPS POST, token)-->     server.py (FastAPI, :8090)
   └ Publicador                                   ├ guarda histórico (metricas.jsonl)
                                                  └ dashboard.html + SSE (tempo real)
```

## Componentes
- `../scripts/treino/treinar_ao_vivo.py` — treino por épocas (MLP sobre embeddings DinoV2), emite métricas.
- `../scripts/treino/publicador.py` — grava JSONL local e faz POST best-effort à VPS.
- `server.py` — FastAPI: recebe POSTs autenticados, serve o dashboard + stream SSE.
- `dashboard.html` — painel de estado do gerador + curvas ao vivo (SVG, sem dependências).

## Rodar o treino publicando no dashboard (máquina local)
```bash
export DASHBOARD_URL=https://iadoc.taberna3d.com.br
export DASHBOARD_TOKEN=<token>      # está em /root/apps/iadoc-dashboard/.env na VPS
python -m scripts.treino.treinar_ao_vivo --epocas 120 --intervalo 0.3
```
Sem `DASHBOARD_URL`, o treino roda igual e só grava o JSONL local.

## Deploy na VPS (já feito — referência)
- App em `/root/apps/iadoc-dashboard/` na porta **8090** (loopback).
- systemd: `iadoc-dashboard.service` (token em `.env`, chmod 600).
- nginx: site `iadoc.taberna3d.com.br` faz proxy → 8090, com `proxy_buffering off` + `proxy_read_timeout 3600s` para o SSE. HTTPS via Let's Encrypt; fronteado por Cloudflare.
- Atualizar: `scp server.py dashboard.html root@vps:/root/apps/iadoc-dashboard/ && ssh root@vps systemctl restart iadoc-dashboard`.

## Notas
- O `Publicador` envia `User-Agent: iadoc-trainer/1.0` — o padrão do urllib é barrado (403) pelo Cloudflare.
- A visualização é pública (só métricas, sem PII). Para restringir, adicionar HTTP Basic Auth no nginx.
