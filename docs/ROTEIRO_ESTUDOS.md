# Roteiro de estudos — detecção de fraude em documentos com IA

Guia dos principais conceitos por trás deste projeto, na ordem sugerida de
estudo (dos pré-requisitos à fronteira). Para cada tópico: **o que é**, **por que
importa aqui** e **palavras-chave** para pesquisar. Ao final há um mapa
conceito → arquivo do projeto e uma lista de consultas prontas para pesquisar.

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
  desfoque, inpainting e composição.
- **Palavras-chave:** OpenCV `cv2`, espaços de cor **RGB/HSV/LAB**, `resize`,
  `warpAffine`/`warpPerspective`, `GaussianBlur`, **Laplaciano** (medida de
  nitidez/foco), **FFT** (frequências de uma imagem), `seamlessClone` (Poisson).

### 0.3 Processamento de imagem aplicado
- **O que é:** técnicas clássicas para simular a captura real e editar regiões.
- **Por que importa:** são a base do simulador de captura (`captura.py`, `ruido.py`)
  e da edição de campos (`inpaint.py`, `textura.py`, `renderizacao_texto.py`).
- **Palavras-chave:** **inpainting** (Telea, Navier-Stokes), **ruído
  Poisson-Gaussiano** de sensor, compressão **JPEG** e seus artefatos, vinheta,
  jitter geométrico/perspectiva, transplante de textura de alta frequência.

---

## Módulo 1 — Machine Learning essencial

### 1.1 Aprendizado supervisionado e classificação
- **O que é:** ensinar um modelo a separar classes (aqui: legítimo × fraude) a
  partir de exemplos rotulados.
- **Por que importa:** é exatamente o objetivo final do projeto.
- **Palavras-chave:** classificação binária, **features** (características),
  rótulo (label), conjuntos de **treino / validação / teste**, **overfitting**.

### 1.2 Métricas e protocolos de avaliação
- **O que é:** como medir se o modelo é bom — e como evitar medir errado.
- **Por que importa:** é onde o projeto tem mais rigor. Além da **AUC**, usamos
  quatro protocolos que existem para não superestimar o desempenho
  (`treinar_classificador.py`):
  - **split agrupado por documento** — todas as variantes de um documento ficam do
    mesmo lado do split, senão o modelo reconhece o documento e a AUC infla;
  - **leave-one-technique-out** — treina sem um tipo de fraude e testa nele;
  - **leave-one-generator-out** — treina sem um dos geradores da mesma técnica e
    testa nele; mede se a diversidade de geradores funcionou;
  - **equilíbrio por origem** — iguala a taxa de fraude entre BID e documentos
    reais, senão a origem da imagem sozinha já prediz parte do rótulo.
- **Palavras-chave:** matriz de confusão, precisão/recall, curva **ROC** e **AUC**
  (0,5 = acaso; 1,0 = perfeito), validação cruzada, **GroupKFold**, **data leakage**.

### 1.3 Modelos lineares
- **O que é:** o classificador mais simples (uma reta/plano separando classes).
- **Por que importa:** a sonda anti-atalho usa **regressão logística** sobre
  estatísticas globais da imagem, e a cabeça padrão da Fase 3 também é logística.
- **Palavras-chave:** regressão logística, `scikit-learn`, normalização
  (StandardScaler), *linear probe*.

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
- **Por que importa:** é **o que o projeto usa hoje**. O `facebook/dinov2-base`
  roda **congelado** e transforma cada imagem num vetor de 768 números
  (`embeddings_dinov2.py`); a única parte treinada é uma **cabeça leve** —
  regressão logística ou um MLP pequeno (`treinar_classificador.py`,
  `treinar_ao_vivo.py`). Funciona bem com dataset pequeno.
- **Palavras-chave:** **foundation model**, **transfer learning**, **embedding**,
  self-supervised learning, **DINOv2**, **linear probe**, features congeladas.

### 2.4 Limite da representação semântica (e o próximo passo)
- **O que é:** embeddings de modelo de fundação são **semânticos** — resumem "o que
  há na imagem", não a microestrutura de pixel.
- **Por que importa:** explica um resultado medido no projeto. A **troca de rosto**
  é detectável (inconsistência estrutural aparece no embedding), mas a **edição de
  texto é quase invisível** — trocar um nome não muda o resumo semântico. A saída
  não é melhorar o gerador de texto: é (a) a camada de dados (checksum de CPF) e
  (b) somar um **canal forense** de baixo nível (Fase B do roadmap).
- **Palavras-chave:** semantic vs low-level features, **NoisePrint/TruFor**, fusão
  RGB + mapa de ruído, arquiteturas híbridas (EdgeDoc).

---

## Módulo 3 — O problema: fraude documental e forense de imagem

### 3.1 Documento legítimo × falsificado
- **O que é:** os tipos de adulteração aplicados na prática.
- **Por que importa:** o projeto foca nas **duas fraudes que fraudadores de fato
  usam** em documentos de identidade: **troca da foto de rosto** e **edição de
  campos** (`geracao_fraude/`).
- **Palavras-chave:** document forgery, ID document fraud, splicing, tampering.

### 3.2 Forense de imagem
- **O que é:** detectar manipulação pelos vestígios que ela deixa.
- **Por que importa:** é o sinal que o modelo deve aprender — e o canal que ainda
  falta no pipeline (ver 2.4).
- **Palavras-chave:** **dupla compressão JPEG**, **PRNU / NoisePrint** (ruído de
  sensor), inconsistências de ruído/iluminação, **ELA** (Error Level Analysis — a
  abordagem antiga, abandonada por ser fácil de burlar).

### 3.3 OCR, campos e orientação
- **O que é:** ler texto da imagem, localizar campos e descobrir a rotação.
- **Por que importa:** `deteccao.py` acha o valor associado a um rótulo (testando as
  4 rotações) e `bid.py` normaliza a orientação dos documentos do BID — decidindo
  pelo **eixo das caixas ground-truth + score de legibilidade do OCR**, e não pelo
  OSD do Tesseract, que se mostrou não confiável.
- **Palavras-chave:** **OCR**, Tesseract, `image_to_data`, `image_to_boxes`,
  detecção de orientação.

---

## Módulo 4 — O conceito central: atalhos e o gap sintético→real

> Este é o coração intelectual do projeto. Vale estudar com calma.

### 4.1 Shortcut learning (aprendizado por atalho)
- **O que é:** quando o modelo aprende uma pista fácil e espúria em vez do que
  realmente importa (ex.: "toda fraude tem compressão diferente" em vez de "esta
  região foi adulterada").
- **Por que importa:** é o problema que a **captura compartilhada** (`captura.py`,
  aplicada igualmente a legítimos e fraudes) e a **sonda anti-atalho** existem para
  combater. Caso concreto medido aqui: as duas origens de dado tinham proporções
  diferentes de fraude, e só saber a origem já dava AUC 0,528 — corrigido
  equilibrando as origens.
- **Palavras-chave:** **shortcut learning**, spurious correlations, dataset bias,
  Clever Hans effect.

### 4.2 Gap sintético→real (Reality Gap / Synthetic Utility Gap)
- **O que é:** um modelo pode ir muito bem em dados sintéticos e falhar no mundo
  real, porque aprendeu a "assinatura do gerador" e não a fraude de verdade.
- **Por que importa:** por isso precisamos de **validação externa** contra fraude
  real (ex.: dataset SIDTD) — o teste que ainda falta. E é a razão de a AUC alta no
  nosso próprio dataset **não** ser o objetivo.
- **Palavras-chave:** domain gap, sim-to-real, generalização, validação externa,
  data leakage.

### 4.3 Diversidade de geradores
- **O que é:** produzir a mesma fraude por vários métodos diferentes, em vez de
  aperfeiçoar um só.
- **Por que importa:** cada gerador deixa uma marca própria; misturando vários, o
  modelo não consegue decorar nenhuma e é obrigado a aprender a fraude. Hoje a
  edição de campos tem dois geradores (renderização clássica e transplante de
  glifos) e o `leave-one-generator-out` mede se a estratégia funcionou.
- **Palavras-chave:** generator diversity, ensemble of manipulations,
  leave-one-generator-out.

### 4.4 Manipulação local × global
- **O que é:** distinção entre fraude que muda uma região (foto, campo) e fraude
  que muda a imagem inteira (reimpressão, recaptura).
- **Por que importa:** a sonda anti-atalho vale para fraudes **locais**; ataques
  **globais** são inerentemente detectáveis por estatística global — foi um dos
  motivos de recortá-los do escopo.
- **Palavras-chave:** local vs global manipulation, forensic features.

---

## Módulo 5 — Geração de dados sintéticos (o que o projeto faz hoje)

### 5.1 Por que gerar fraude
- **O que é:** como quase não existe fraude real rotulada disponível, geramos
  fraude a partir de documentos legítimos.
- **Palavras-chave:** data augmentation, synthetic data, class balancing.

### 5.2 As duas técnicas e seus geradores
- **Troca de foto de rosto** (`troca_foto.py`): detecção e malha de 478 landmarks
  via MediaPipe (`rosto.py`), alinhamento por transformação de similaridade,
  máscara pela silhueta do rosto, harmonização de cor em LAB e blend de Poisson.
  Variante "colagem simples" mantida como par fácil × difícil.
- **Edição de campos** (`edicao_campos.py`, `gerar_dataset_bid.py`): remove só os
  traços da tinta (`inpaint.remover_tinta`) e reescreve o valor por dois caminhos —
  renderização com fonte do sistema ou **transplante dos glifos do próprio
  documento** (`glifos.py`).
- **Palavras-chave:** face swap/splicing, scene text editing, landmarks faciais,
  Poisson blending, transferência de cor Reinhard.

### 5.3 Paridade de pipeline e rastreabilidade
- **O que é:** toda amostra — fraude **e** legítima — passa pelo mesmo simulador de
  captura (`captura.py` via `saida.py`), e cada uma grava um manifesto JSON com
  origem, técnica, gerador e parâmetros.
- **Por que importa:** a paridade garante que a única diferença sistemática entre as
  classes seja a manipulação; o manifesto sustenta os protocolos de avaliação e a
  auditoria do dataset.
- **Palavras-chave:** reprodutibilidade, metadados, auditoria de dataset,
  pipeline parity.

---

## Módulo 6 — Modelos generativos (a fronteira que estamos explorando)

> Nada disto está em produção no pipeline: são os candidatos avaliados para somar
> geradores (diversidade) e realismo.

### 6.1 Inpainting generativo
- **O que é:** IA que reconstrói regiões apagadas de forma realista.
- **Por que importa:** candidato para reconstruir o fundo (guilhoché) melhor que os
  métodos clássicos. **LaMa** é forte em padrões periódicos, mas o ambiente oficial
  dele exige uma versão antiga do PyTorch — rodá-lo aqui exige porte para cu128.
- **Palavras-chave:** **LaMa**, image inpainting, Fourier convolutions.

### 6.2 Modelos de difusão e edição de texto em imagem
- **O que é:** modelos que geram/editam imagem (e texto dentro dela) preservando estilo.
- **Por que importa:** seria o terceiro gerador de `edicao_campos`. Regra que a
  literatura impõe: editar **apenas a região do campo** e recompor sobre os pixels
  originais — reprocessar o documento inteiro injeta uma assinatura global que vira
  atalho.
- **Palavras-chave:** **diffusion models**, Stable Diffusion inpainting, ControlNet,
  **scene text editing** (AnyText, TextCtrl, FLUX-Text), quantização NF4.

### 6.3 Face swap neural e harmonização
- **O que é:** troca de rosto por rede neural e ajuste de cor/iluminação aprendido.
- **Por que importa:** o **InSwapper** seria o segundo gerador de troca de foto — o
  que tornaria o `leave-one-generator-out` informativo para a única fraude que o
  detector hoje enxerga. **PCT-Net** substituiria o Poisson na harmonização.
- **Palavras-chave:** **InSwapper / InsightFace**, face swapping, InstantID,
  **PCT-Net**, deep image harmonization.

---

## Módulo 7 — Ferramentas e infraestrutura

- **PyTorch + GPU:** framework de deep learning; entender **CUDA** e compatibilidade
  de arquitetura (a RTX 5050 é Blackwell/sm_120 e exige **cu128**).
- **Persistência:** **MongoDB** local guarda metadados do dataset e métricas de
  treino (`scripts/db/`). Nenhuma imagem entra no banco e os identificadores de
  documentos reais são **pseudonimizados** por hash.
- **Acompanhamento:** o treino publica métricas por época via HTTP para um painel
  web (`dashboard/`), que as transmite ao navegador por **SSE**.
- **Ambiente:** ambientes virtuais (`venv`), `pip`/`requirements.txt`, Git.
- **Palavras-chave:** PyTorch, CUDA, compute capability, MongoDB, FastAPI,
  server-sent events, virtualenv, versionamento.

---

## Módulo 8 — Ética, privacidade e LGPD

- **O que é:** documentos de identidade são dados pessoais sensíveis (a foto do
  rosto é dado biométrico).
- **Por que importa:** define regras práticas do projeto — documentos reais nunca
  saem da máquina nem são versionados, o dataset **BID** (sintético) faz o volume,
  e o banco recebe apenas metadados pseudonimizados.
- **Palavras-chave:** **LGPD**, PII, dado biométrico, consentimento,
  pseudonimização × anonimização, dados sintéticos.

---

## Mapa conceito → arquivo do projeto

| Conceito | Onde ver no código |
|---|---|
| Espaços de cor, Laplaciano, FFT | `scripts/comum/captura.py`, `scripts/avaliacao/sanity_atalho.py` |
| Simulador de captura (paridade de pipeline) | `scripts/comum/captura.py`, `scripts/geracao_fraude/saida.py` |
| Ruído de sensor Poisson-Gaussiano | `scripts/comum/ruido.py` |
| Inpainting (remoção só da tinta) | `scripts/comum/inpaint.py` |
| Harmonização de micro-textura | `scripts/comum/textura.py` |
| Renderização de texto e ajuste de nitidez | `scripts/comum/renderizacao_texto.py` |
| Transplante de glifos reais | `scripts/comum/glifos.py` |
| Landmarks faciais (MediaPipe) | `scripts/comum/rosto.py` |
| OCR, detecção de campo e orientação | `scripts/comum/deteccao.py`, `scripts/comum/bid.py` |
| Valores plausíveis (Faker) e fontes | `scripts/comum/valores.py` |
| Checksum de CPF | `scripts/comum/validacao.py` |
| Técnicas de fraude | `scripts/geracao_fraude/troca_foto.py`, `edicao_campos.py` |
| Orquestração do dataset | `scripts/geracao_fraude/gerar_dataset_bid.py`, `gerar_dataset_reais.py` |
| Manifesto / rastreabilidade | `scripts/geracao_fraude/manifesto.py` |
| Sonda anti-atalho (estatísticas globais) | `scripts/avaliacao/sanity_atalho.py` |
| Embeddings DinoV2 | `scripts/features/embeddings_dinov2.py` |
| Cabeça de classificação e protocolos de avaliação | `scripts/treino/treinar_classificador.py` |
| Treino por épocas com métricas ao vivo | `scripts/treino/treinar_ao_vivo.py`, `publicador.py` |
| Persistência de metadados e métricas | `scripts/db/mongo.py`, `scripts/db/ingestao.py` |
| Painel de acompanhamento | `dashboard/server.py`, `dashboard/dashboard.html` |
| Validação de dados (MRZ, QR) | `scripts/validacao_dados/` (a implementar) |
| Canal forense (NoisePrint/TruFor) | não implementado — Fase B do roadmap |

---

## Ordem sugerida de estudo

1. **Módulo 0** (Python científico + OpenCV) — base prática.
2. **Módulo 1** (ML essencial + métricas) — para entender treino, AUC e os
   protocolos de avaliação que o projeto usa.
3. **Módulo 4** (atalhos e gap sintético→real) — o conceito que guia todo o design.
4. **Módulo 3** (forense de imagem) + **Módulo 5** (geração) — o que já está pronto.
5. **Módulo 2** (deep learning, embeddings e o limite da representação) — é o que
   está rodando hoje e define o próximo passo de arquitetura.
6. **Módulos 6–8** conforme a necessidade (fronteira, infra, ética).

---

## Consultas de estudo

Termos prontos para pesquisar (Google, YouTube, Perplexity). Quando o material
relevante só existe em inglês, a consulta está em inglês — marcado com **[EN]**.

### Módulo 0 — Fundamentos
- `numpy manipulação de imagens como matriz tutorial`
- `OpenCV Python curso espaços de cor BGR RGB HSV LAB`
- `variância do laplaciano detectar imagem desfocada`
- `transformada de Fourier em imagens explicação visual`
- `artefatos de compressão JPEG blocagem explicação`
- **[EN]** `image inpainting Telea vs Navier-Stokes OpenCV`
- **[EN]** `Poisson image editing seamless cloning explained`

### Módulo 1 — Machine Learning essencial
- `curva ROC e AUC explicação intuitiva`
- `overfitting e underfitting explicação com exemplos`
- `validação cruzada k-fold quando usar`
- `regressão logística scikit-learn na prática`
- **[EN]** `grouped cross validation StratifiedGroupKFold why` — o split que usamos
  para variantes do mesmo documento não vazarem entre treino e teste
- **[EN]** `data leakage machine learning examples how to detect`
- **[EN]** `class imbalance effect on ROC AUC`

### Módulo 2 — Deep Learning, embeddings e modelos de fundação
> É o núcleo do que o projeto usa hoje: DinoV2 **congelado** gerando embeddings +
> uma cabeça leve treinada por cima.
- `o que são embeddings em deep learning explicação`
- `transfer learning explicação prática`
- `redes neurais convolucionais CNN explicação`
- **[EN]** `DINOv2 explained self-supervised vision features`
- **[EN]** `linear probe vs fine-tuning frozen features` — por que treinamos só a
  cabeça em vez de ajustar o DinoV2 inteiro
- **[EN]** `vision transformer ViT explained patch embeddings`
- **[EN]** `why foundation model embeddings miss low-level forensic cues` — explica
  por que a edição de texto fica quase invisível para o DinoV2

### Módulo 3 — Fraude documental e forense de imagem
- `forense digital de imagens detecção de manipulação`
- **[EN]** `Noiseprint camera model fingerprint forgery detection`
- **[EN]** `TruFor image forgery localization`
- **[EN]** `PRNU sensor pattern noise explained`
- **[EN]** `double JPEG compression detection forensics`
- **[EN]** `identity document presentation attack detection survey`

### Módulo 4 — Atalhos e gap sintético→real (o conceito central)
- **[EN]** `shortcut learning in deep neural networks` — artigo de referência
  (Geirhos et al., 2020)
- **[EN]** `spurious correlations machine learning Clever Hans`
- **[EN]** `synthetic to real domain gap generalization`
- **[EN]** `leave-one-out generalization test forgery detector`
- **[EN]** `generator diversity synthetic forgery detection`
- `viés de dataset explicação exemplos`

### Módulo 5 — Geração de dados sintéticos
- **[EN]** `Self-Blended Images deepfake detection CVPR 2022` — a ideia de paridade
  de pipeline que o nosso `captura.py` aplica
- **[EN]** `synthetic training data forensic detector pitfalls`
- `data augmentation boas práticas`
- **[EN]** `SIDTD dataset identity documents` — a base de fraude real que falta
- **[EN]** `MediaPipe face landmarker 478 landmarks tasks API`

### Módulo 6 — Modelos generativos (fronteira)
- **[EN]** `diffusion models explained step by step`
- **[EN]** `Stable Diffusion inpainting how it works`
- **[EN]** `scene text editing diffusion AnyText TextCtrl`
- **[EN]** `LaMa inpainting Fourier convolutions`
- **[EN]** `InSwapper InsightFace face swap how it works`
- **[EN]** `deep image harmonization PCT-Net`

### Módulo 7 — Ferramentas e infraestrutura
- `PyTorch curso introdução tensores`
- **[EN]** `CUDA compute capability GPU compatibility PyTorch`
- `MongoDB básico coleções documentos índices`
- `FastAPI tutorial português`
- **[EN]** `server-sent events SSE vs websockets`

### Módulo 8 — Ética e privacidade
- `LGPD dados pessoais sensíveis o que diz a lei`
- `LGPD dados biométricos tratamento`
- `pseudonimização versus anonimização LGPD ANPD`

### Fontes canônicas (para ir à origem)
| Tema | Referência |
|---|---|
| Revisão de PAD em documentos | arXiv **2511.06056** |
| Shortcut learning | arXiv **2004.07780** (Geirhos et al.) |
| Self-Blended Images | CVPR 2022 — arXiv **2204.08376** |
| DINOv2 | arXiv **2304.07193** (Meta AI) |
| TruFor / Noiseprint++ | arXiv **2212.10957** |
| LaMa (inpainting) | arXiv **2109.07161** |
| Detectores e artefatos globais | arXiv **2602.00192** |
| Documentação oficial | `scikit-learn.org`, `pytorch.org`, `docs.opencv.org`, `huggingface.co/docs` |
