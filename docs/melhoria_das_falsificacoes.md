# Como melhorar o realismo das falsificações geradas

Documento de pesquisa — 2026-07-03
Projeto: `analise_ia_doc`
Motivação: as 6 técnicas atuais produzem artefatos "estranhos", com risco de o
classificador (DinoV2) aprender a assinatura do *nosso gerador* em vez de fraude
real — o "Synthetic Utility Gap" descrito na documentação técnica (Seção 7.2).

---

## 1. Princípio central (ler antes de qualquer mudança)

O objetivo **não** é gerar a falsificação visualmente mais bonita possível. O
objetivo é que a *distribuição de artefatos forenses* das nossas fraudes se
pareça com a de fraudes reais, para o detector generalizar. Duas consequências:

1. **Diversidade de geradores importa mais que perfeição de um gerador.** A
   literatura de PAD (arXiv 2511.06056, já citada na nossa doc) mostra que a
   principal mitigação do Synthetic Utility Gap é (a) diversificar ao máximo os
   métodos de geração e (b) **validar sempre contra uma técnica de fraude que
   NÃO foi usada no treino**. Um único gerador "perfeito" ainda deixa uma
   assinatura única que o modelo decora.

2. **Nem todo artefato feio é ruim — mas os nossos são do tipo ruim.** Os
   artefatos atuais (dithering Bayer 4×4 visível, moiré senoidal, colagem
   retangular de rosto) são *sistemáticos e sintéticos*: aparecem em toda
   imagem, no mesmo padrão, e não existem em fraude real. É exatamente esse tipo
   de sinal constante que uma rede aprende como atalho. Precisamos substituí-los
   por artefatos que ou (a) vêm de um processo físico real, ou (b) vêm de
   modelos generativos modernos cujos traços se aproximam do que um fraudador de
   verdade usaria hoje.

Referência direta do estado da arte: o dataset **FantasyID** (benchmark do
desafio DeepID/ICCV 2025) gera fraude com **face-swap por rede neural
(InSwapper, FaceDancer)** e **edição de texto por difusão (DiffSTE,
TextDiffuser2)**, e — crucialmente — **imprime e recaptura fisicamente** os
documentos com 3 dispositivos diferentes. É esse padrão que devemos seguir.

---

## 2. Diagnóstico por técnica (por que sai estranho hoje)

| Técnica | O que o código faz hoje | Por que fica irreal |
|---|---|---|
| `troca_foto` | Recorta a caixa do rosto, **redimensiona pro retângulo alvo** e cola; blend por `seamlessClone`/Poisson + transferência de cor Reinhard LAB | Sem alinhar landmarks (olhos/boca) nem warp geométrico → rosto esticado/torto. Poisson espalha matiz (*hue bleeding*) sobre o fundo texturizado do documento. Ajuste de nitidez só borra, nunca modela upscaling. |
| `edicao_campos` | OCR acha o campo, apaga o texto e **redesenha com fonte do sistema** | A fonte nunca bate com a oficial; texto sai limpo demais (sem textura de impressão, sem jitter de baseline). Removedor deixa borrão/retângulo. Além disso, só gerou 53 imagens → o detector de rótulo por OCR falha na maioria dos documentos. |
| `crop_and_move` | Recorta região, cola em outro lugar com *feather*, preenche buraco com mediana local + ruído gaussiano | O preenchimento por mediana cria manchas chapadas sem estrutura; sem harmonização, a região movida mantém iluminação da origem. |
| `digito_verificador` | Altera dígitos numéricos | É o caso mais defensável (ataque à camada de dados), mas herda o mesmo problema de renderização de texto do `edicao_campos`. |
| `recaptura_tela` | Moiré **senoidal sintético** + shift de canal + gradiente de reflexo + ruído + cascata JPEG | Moiré real vem da batida entre a grade de subpixels da tela e a amostragem da câmera; a senoide fixa parece artificial e é constante demais. |
| `reimpressao` | **Dithering ordenado Bayer 4×4** + textura de papel + vinheta + ruído + cascata JPEG | Impressora real usa meio-tom CMYK aglomerado/difusão de erro em alta DPI (invisível a olho nu). O padrão Bayer 4×4 é grosseiro e obviamente sintético. |

---

## 3. Recomendações por técnica

### 3.1 `troca_foto` (a mais crítica)
- **Trocar a colagem retangular por um face-swapper neural**: `InSwapper`
  (InsightFace, `inswapper_128.onnx`) ou `SimSwap`/`FaceDancer`. Eles alinham
  landmarks, fazem warp de pose e preservam iluminação — exatamente os mesmos
  usados pelo FantasyID. Roda em ONNXRuntime; leve.
- **Substituir o Poisson por harmonização por rede** (deep image harmonization):
  `Harmonizer`, `PCT-Net` ou a toolbox `libcom`. Poisson "vaza" matiz e não
  corrige textura; harmonização aprendida ajusta cor/iluminação da região colada
  ao entorno sem o smear. Ver "Deep Image Composition Meets Image Forgery".
- **Manter a variante "evidente" (colagem simples)** de propósito — o par
  fácil/difícil continua válido. Só melhore o lado "sutil".

### 3.2 `edicao_campos` + `digito_verificador`
- **Editar texto por inpainting de difusão** condicionado ao estilo:
  `TextDiffuser-2` ou `DiffSTE` (usados no FantasyID), ou `AnyText`. Eles
  preservam fonte, cor e textura de fundo muito melhor que "apagar + redesenhar".
- Se difusão for pesada demais no começo: ao menos **extrair a fonte real**
  (casar glyphs recortados do próprio documento) e aplicar **degradação de
  impressão** (leve blur + ruído + recompressão) sobre o texto novo, para não
  ficar mais nítido que o resto.
- **Corrigir a cobertura**: hoje só 53 imagens porque o detector de rótulo por
  OCR falha. Vale um fallback por posição/template quando o OCR não acha o campo,
  ou trocar o OCR (EasyOCR já está no `requirements.txt`).

### 3.3 `crop_and_move`
- **Preencher o buraco com inpainting** (LaMa, ou inpainting de difusão) em vez
  de mediana local chapada.
- Passar a região movida pela **mesma harmonização** da 3.1.

### 3.4 `recaptura_tela` e `reimpressao` — a mudança de maior impacto
Estas duas classes são as que mais sofrem com simulação sintética. A recomendação
forte da literatura (FantasyID, e o survey de PAD) é **capturar fisicamente**:
- **Reimpressão real**: imprimir um lote de documentos (legítimos e já
  manipulados) numa impressora comum e **refotografar com celular**. Isso gera o
  meio-tom, o ruído de papel e a cascata de compressão *reais*, impossíveis de
  imitar bem por código.
- **Recaptura de tela real**: exibir o documento num monitor e fotografar com
  celular, variando ângulo/distância — gera moiré e reflexo especular reais.
- Se o volume físico for inviável agora, ao menos **trocar o dithering Bayer por
  meio-tom realista** (difusão de erro Floyd–Steinberg em alta resolução, depois
  downscale) e **substituir o moiré senoidal** por *moiré pattern augmentation*
  aprendido/físico (ver "Simulated Physical Spoofing Clues", arXiv 2404.08450, e
  "Moiré Attack", arXiv 2110.10444). Mas o ganho de capturar de verdade é muito
  maior.

---

## 4. Mitigação obrigatória do Synthetic Utility Gap
Independente de quão realistas fiquem as fraudes:
1. **Validação cruzada por gerador**: separar uma técnica (ou um gerador, ex.:
   InSwapper) só para teste, treinar nas outras. Se a acurácia despencar, o
   modelo está decorando o gerador, não aprendendo fraude.
2. **Misturar geradores** na mesma classe semântica (ex.: face-swap por
   InSwapper *e* SimSwap *e* colagem clássica) para diluir a assinatura.
3. **Incluir um punhado de amostras físicas reais** (impressão/tela) como
   conjunto de validação de generalização — é o teste mais honesto que temos.
4. Validar também contra um dataset público (MIDV-2020/SIDTD/FantasyID) para
   medir o "Reality Gap".

---

## 5. Nuance importante sobre o DinoV2
O DinoV2 trabalha com *embeddings semânticos* de alto nível e é relativamente
**insensível a artefatos de baixo nível** (ruído, compressão). Isso tem dois
lados: (a) ele pode simplesmente *ignorar* nossos artefatos sintéticos feios — o
que reduz um pouco o risco — mas (b) por isso mesmo ele também pode ignorar
sinais forenses sutis de fraude real. É a justificativa da **Fase B** da nossa
doc (canal de NoisePrint/TruFor junto do RGB): o sinal de forense de ruído é o
que carrega os artefatos de splicing/recaptura. Ou seja: melhorar as
falsificações **e** adicionar o canal de ruído são complementares, não
alternativos.

---

## 6. Restrição prática: a VPS não tem GPU
Os métodos neurais (face-swap, inpainting de difusão, harmonização) rodam
**muito melhor com GPU**. A VPS atual (2 vCPU, 7,8 GB RAM, sem GPU) roda
InSwapper/ONNX em CPU de forma lenta mas viável; difusão (TextDiffuser2) em CPU é
impraticável para lote grande. Opções: (a) gerar o dataset uma vez numa GPU
alugada por hora e copiar o resultado; (b) começar pelo que roda em CPU
(InSwapper + harmonização + meio-tom realista + captura física) e deixar difusão
para depois.

---

## 7. Ordem sugerida de implementação
1. `troca_foto`: InSwapper + harmonização (maior impacto, roda em CPU).
2. Captura física de um lote para `reimpressao`/`recaptura_tela`.
3. Trocar Bayer→Floyd–Steinberg e moiré senoidal→augmentation físico (fallback).
4. `edicao_campos`: casar fonte real + degradação; difusão quando houver GPU.
5. `crop_and_move`: inpainting LaMa + harmonização.
6. Montar o protocolo de validação cruzada por gerador (Seção 4) antes de treinar.

---

## 8. Fontes
- FantasyID — A dataset for detecting digital manipulations of ID-documents (arXiv 2507.20808): https://arxiv.org/abs/2507.20808
- DeepID Challenge (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025W/DeepID/papers/Korshunov_DeepID_Challenge_of_Detecting_Synthetic_Manipulations_in_ID_Documents_ICCVW_2025_paper.pdf
- EdgeDoc (arXiv 2508.16284): https://arxiv.org/html/2508.16284v1
- Identity Card Presentation Attack Detection: A Systematic Review (arXiv 2511.06056): https://arxiv.org/html/2511.06056
- AIForge-Doc — diffusion inpainting em documentos (arXiv 2602.20569): https://arxiv.org/html/2602.20569v1
- TGIF — Text-Guided Inpainting Forgery (arXiv 2407.11566): https://arxiv.org/abs/2407.11566
- Deep Image Composition Meets Image Forgery (arXiv 2404.02897): https://arxiv.org/html/2404.02897v1
- Deep Image Blending (arXiv 1910.11495): https://arxiv.org/pdf/1910.11495
- Joint Physical-Digital Facial Attack Detection / Simulated Physical Spoofing Clues (arXiv 2404.08450): https://arxiv.org/pdf/2404.08450
- Moiré Attack (arXiv 2110.10444): https://arxiv.org/pdf/2110.10444
- FakeIDet2 — Privacy-Aware Detection of Fake Identity Documents (arXiv 2508.11716): https://arxiv.org/html/2508.11716v2
