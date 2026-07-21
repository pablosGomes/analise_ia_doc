# Roteiro de estudos — detecção de fraude em documentos com IA

Guia dos principais conceitos por trás deste projeto, na ordem sugerida de
estudo (dos pré-requisitos à fronteira). Para cada tópico: **o que é**, **por que
importa aqui** e **palavras-chave** para pesquisar. Ao final há um mapa
conceito → arquivo do projeto.

> Como usar: siga os módulos em ordem. Não precisa dominar tudo antes de avançar;
> a ideia é ter o vocabulário e a intuição para entender e evoluir o código.

---

## Módulo 0 — Fundamentos (base para tudo)

### 0.1 Python científico e imagens como matrizes
- **O que é:** uma imagem é uma matriz de números (altura × largura × canais de
  cor). Manipular imagem = manipular arrays.
- **Por que importa:** todo o pipeline (`scripts/comum/`) trabalha assim.
- **Palavras-chave:** `numpy` (arrays, slicing, broadcasting), imagem BGR/RGB,
  `dtype uint8` vs `float32`.

### 0.2 Visão computacional com OpenCV
- **O que é:** biblioteca para ler, transformar e analisar imagens.
- **Por que importa:** usamos para rotação, redimensionamento, conversão de cor,
  desfoque, inpainting, detecção de rosto.
- **Palavras-chave:** OpenCV `cv2`, espaços de cor **RGB/HSV/LAB**, `resize`,
  `warpAffine`/`warpPerspective`, `GaussianBlur`, **Laplaciano** (medida de
  nitidez/foco), **FFT** (frequências de uma imagem).

### 0.3 Processamento de imagem aplicado
- **O que é:** técnicas clássicas para simular artefatos físicos.
- **Por que importa:** são a base das nossas técnicas de fraude e do simulador de
  captura (`captura.py`, `halftone.py`, `ruido.py`).
- **Palavras-chave:** **inpainting** (Telea, Navier-Stokes), **halftone/dithering**
  (Bayer, Floyd-Steinberg), **moiré**, **ruído Poisson-Gaussiano** de sensor,
  compressão **JPEG** e seus artefatos.

---

## Módulo 1 — Machine Learning essencial

### 1.1 Aprendizado supervisionado e classificação
- **O que é:** ensinar um modelo a separar classes (aqui: legítimo × fraude) a
  partir de exemplos rotulados.
- **Por que importa:** é exatamente o objetivo final do projeto.
- **Palavras-chave:** classificação binária, **features** (características),
  rótulo (label), conjuntos de **treino / validação / teste**, **overfitting**.

### 1.2 Métricas de avaliação
- **O que é:** como medir se o modelo é bom.
- **Por que importa:** usamos **AUC** na sonda anti-atalho (`sanity_atalho.py`).
- **Palavras-chave:** matriz de confusão, precisão/recall, curva **ROC** e **AUC**
  (0,5 = acaso; 1,0 = perfeito), validação cruzada (**cross-validation**).

### 1.3 Modelos lineares
- **O que é:** o classificador mais simples (uma reta/plano separando classes).
- **Por que importa:** a sonda anti-atalho usa **regressão logística** sobre
  estatísticas globais da imagem.
- **Palavras-chave:** regressão logística, `scikit-learn`, normalização
  (StandardScaler).

---

## Módulo 2 — Deep Learning e modelos de fundação

### 2.1 Redes neurais e CNNs
- **O que é:** modelos que aprendem as próprias features a partir dos dados.
- **Por que importa:** é a base dos modelos modernos de visão.
- **Palavras-chave:** rede neural, **CNN** (convolução), backpropagation, **GPU**.

### 2.2 Transformers e Vision Transformers (ViT)
- **O que é:** arquitetura que virou padrão em visão e linguagem.
- **Por que importa:** o DinoV2 é um ViT.
- **Palavras-chave:** Transformer, atenção (attention), **ViT**, patch embeddings.

### 2.3 Modelos de fundação, transfer learning e embeddings
- **O que é:** modelos grandes pré-treinados que geram uma "impressão digital"
  (embedding) da imagem; reaproveitados em tarefas novas com pouco dado.
- **Por que importa:** o plano da Fase 3 é usar **DinoV2** para extrair embeddings
  e treinar uma cabeça leve por cima.
- **Palavras-chave:** **foundation model**, **transfer learning**, **embedding**,
  self-supervised learning, **DINOv2**, cabeça de classificação (linear probe).

---

## Módulo 3 — O problema: fraude documental e forense de imagem

### 3.1 Documento legítimo × falsificado
- **O que é:** os tipos de adulteração (troca de foto, edição de campo, etc.).
- **Por que importa:** cada tipo vira uma técnica de geração (`geracao_fraude/`).
- **Palavras-chave:** document forgery, ID document fraud, splicing, tampering.

### 3.2 Forense de imagem
- **O que é:** detectar manipulação pelos vestígios que ela deixa.
- **Por que importa:** é o sinal que o modelo deve aprender.
- **Palavras-chave:** **dupla compressão JPEG**, **PRNU / noiseprint** (ruído de
  sensor), inconsistências de ruído/iluminação, **ELA** (Error Level Analysis — a
  abordagem antiga, que abandonamos por ser fácil de burlar).

### 3.3 OCR e orientação
- **O que é:** ler texto de imagem e descobrir a rotação do documento.
- **Por que importa:** `deteccao.py` e `bid.py` usam OCR para achar/normalizar campos.
- **Palavras-chave:** **OCR**, Tesseract, **OSD** (orientation detection).

---

## Módulo 4 — O conceito central: atalhos e o gap sintético→real

> Este é o coração intelectual do projeto. Vale estudar com calma.

### 4.1 Shortcut learning (aprendizado por atalho)
- **O que é:** quando o modelo aprende uma pista fácil e espúria em vez do que
  realmente importa (ex.: "toda fraude tem compressão diferente" em vez de "esta
  região foi adulterada").
- **Por que importa:** é o problema que a nossa "captura compartilhada" e a
  "sonda anti-atalho" existem para combater.
- **Palavras-chave:** **shortcut learning**, spurious correlations, dataset bias,
  Clever Hans effect.

### 4.2 Gap sintético→real (Reality Gap / Synthetic Utility Gap)
- **O que é:** um modelo pode ir muito bem em dados sintéticos e falhar no mundo
  real, porque aprendeu a "assinatura do gerador" e não a fraude de verdade.
- **Por que importa:** por isso precisamos de **validação externa** contra fraude
  real (ex.: dataset SIDTD) — o teste que ainda falta.
- **Palavras-chave:** domain gap, sim-to-real, generalização, validação externa,
  data leakage.

### 4.3 Manipulação local × global
- **O que é:** distinção entre fraude que muda uma região (foto, campo) e fraude
  que muda a imagem inteira (reimpressão, recaptura).
- **Por que importa:** a sonda anti-atalho vale para fraudes **locais**; ataques
  **globais** são inerentemente detectáveis por estatística global.
- **Palavras-chave:** local vs global manipulation, forensic features.

---

## Módulo 5 — Geração de dados sintéticos (o que o projeto faz hoje)

### 5.1 Por que gerar fraude
- **O que é:** como quase não existe fraude real rotulada disponível, geramos
  fraude a partir de documentos legítimos.
- **Palavras-chave:** data augmentation, synthetic data, class balancing.

### 5.2 As técnicas do projeto
- **Palavras-chave por técnica:** face swap/splicing (`troca_foto`), text editing
  (`edicao_campos`), checksum/**dígito verificador do CPF** (`digito_verificador`),
  crop-and-move (`crop_and_move`), moiré de recaptura (`recaptura_tela`), halftone
  de reimpressão (`reimpressao`).

### 5.3 Rastreabilidade
- **O que é:** cada amostra gerada guarda um manifesto (JSON) com origem e parâmetros.
- **Palavras-chave:** reprodutibilidade, metadados, auditoria de dataset.

---

## Módulo 6 — Modelos generativos (a fronteira que estamos explorando)

### 6.1 Inpainting generativo
- **O que é:** IA que reconstrói regiões apagadas de forma realista.
- **Por que importa:** usamos **LaMa** para reconstruir fundo em `crop_and_move`.
- **Palavras-chave:** **LaMa**, image inpainting, Fourier convolutions.

### 6.2 Modelos de difusão e edição de texto em imagem
- **O que é:** modelos que geram/editam imagem (e texto dentro dela) preservando estilo.
- **Por que importa:** é o próximo salto de realismo para `edicao_campos`.
- **Palavras-chave:** **diffusion models**, Stable Diffusion, ControlNet,
  **scene text editing** (TextCtrl, AnyText, DiffSTE).

### 6.3 Face swap
- **Palavras-chave:** **InSwapper / InsightFace**, face swapping, morphing.

---

## Módulo 7 — Ferramentas e infraestrutura

- **PyTorch + GPU:** framework de deep learning; entender **CUDA**, compatibilidade
  de arquitetura de GPU (ex.: Blackwell exige CUDA 12.8/cu128).
- **Ambiente:** ambientes virtuais (`venv`), `pip`/`requirements.txt`, Git.
- **Palavras-chave:** PyTorch, CUDA, compute capability, virtualenv, versionamento.

---

## Módulo 8 — Ética, privacidade e LGPD

- **O que é:** documentos de identidade são dados pessoais sensíveis.
- **Por que importa:** regras de consentimento, uso de dados **sintéticos** (BID)
  para evitar exposição de PII, e o cuidado de nunca versionar dados reais.
- **Palavras-chave:** **LGPD**, PII, consentimento, anonimização, dados sintéticos.

---

## Mapa conceito → arquivo do projeto

| Conceito | Onde ver no código |
|---|---|
| Espaços de cor, Laplaciano, FFT | `scripts/comum/captura.py`, `sanity_atalho.py` |
| Inpainting (clássico + LaMa) | `scripts/comum/inpaint.py` |
| Halftone / dithering | `scripts/comum/halftone.py` |
| Ruído de sensor / moiré | `scripts/comum/ruido.py` |
| OCR / detecção de campo / orientação | `scripts/comum/deteccao.py`, `bid.py` |
| Checksum de CPF | `scripts/comum/validacao.py` |
| Técnicas de fraude | `scripts/geracao_fraude/*.py` |
| Paridade de pipeline (anti-atalho) | `scripts/comum/captura.py`, `saida.py` |
| Métrica AUC / sonda de atalho | `scripts/avaliacao/sanity_atalho.py` |
| Embeddings / DinoV2 (Fase 3) | `scripts/features/` (a implementar) |

---

## Ordem sugerida de estudo

1. **Módulo 0** (Python científico + OpenCV) — base prática.
2. **Módulo 1** (ML essencial + métricas) — para entender treino e AUC.
3. **Módulo 4** (atalhos e gap sintético→real) — o conceito que guia todo o design.
4. **Módulo 3** (forense de imagem) + **Módulo 5** (geração) — o que já está pronto.
5. **Módulo 2** (deep learning + DinoV2) — para a Fase 3.
6. **Módulos 6–8** conforme a necessidade (fronteira, infra, ética).
