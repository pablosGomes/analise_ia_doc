"""Adaptador do BID Dataset (documentos brasileiros públicos, dados pessoais
SINTÉTICOS) para o pipeline de geração de fraude.

Diferente dos documentos legítimos reais (fotos de celular, sem rótulos de
campo), o BID fornece o ground-truth de OCR de cada campo (`<id>_gt_ocr.txt`):
a caixa exata e a transcrição de cada trecho de texto. Isso elimina a
dependência do OCR frágil de `deteccao.py` na técnica `edicao_campos` e permite
gerar em escala amostras precisas de edição de campo.

Responsabilidades deste módulo:
    - `ler_gt_ocr`: parse do .txt (encoding cp1252) em caixas + texto.
    - `detectar_rotacao` / `corrigir_orientacao`: os documentos do BID vêm em
      orientações variadas (0/90/180/270). Como `renderizacao_texto.desenhar_texto`
      escreve texto na horizontal, a imagem e as caixas são normalizadas para "de
      pé" antes da edição, senão o texto reescrito ficaria virado em relação à
      vizinhança — um artefato que não corresponderia a uma fraude real.
    - `classificar_campos`: identifica por CONTEÚDO (regex + vizinhança de
      rótulo) quais caixas são dados pessoais editáveis (nome, filiação, data,
      cpf, registro), excluindo rótulos e cabeçalhos fixos do documento.
    - `amostrar` / `carregar_documento`: amostragem balanceada e carregamento.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import pytesseract

# Importa deteccao pelo efeito colateral de configurar o binário do Tesseract no
# Windows (idempotente). bid.py depende do OSD do tesseract para a orientação.
from scripts.comum import deteccao as _deteccao
_deteccao._configurar_tesseract_windows()

# Classes do BID mais ricas em campos de texto editáveis (frente de CNH/CPF e
# RG aberto). Verso/RG_Frente têm poucos campos e são pulados por padrão.
CLASSES_RICAS = ("CNH_Frente", "CNH_Aberta", "CPF_Frente", "RG_Aberto")

RE_DATA = re.compile(r"^\d{2}/\d{2}/\d{4}$")
RE_CPF = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
RE_NUM_LONGO = re.compile(r"^\d{7,}$")

# Tokens que marcam um trecho como RÓTULO/cabeçalho fixo (nunca dado editável).
_STOP = {
    "REPUBLICA", "FEDERATIVA", "BRASIL", "MINISTERIO", "CIDADES", "DEPARTAMENTO",
    "NACIONAL", "TRANSITO", "CARTEIRA", "HABILITACAO", "NOME", "DOC", "IDENTIDADE",
    "ORG", "EMISSOR", "CPF", "DATA", "NASCIMENTO", "FILIACAO", "PERMISSAO", "CAT",
    "HAB", "VALIDADE", "REGISTRO", "VALIDA", "TERRITORIO", "ESTADO", "SECRETARIA",
    "SEGURANCA", "PUBLICA", "INSTITUTO", "IDENTIFICACAO", "PROIBIDO", "PLASTIFICAR",
    "ASSINATURA", "DIRETOR", "TITULAR", "POLEGAR", "DIREITO", "NATURALIDADE",
    "EXPEDICAO", "GERAL", "ORIGEM", "FAZENDA", "RECEITA", "FEDERAL", "CADASTRO",
    "PESSOAS", "FISICAS", "NUMERO", "INSCRICAO", "LV", "FL", "NAS", "CMC", "SEDE",
}


@dataclass(frozen=True)
class CampoBID:
    tipo: str            # nome | filiacao | data | cpf | registro
    x: int
    y: int
    w: int
    h: int
    texto_original: str

    def como_tupla(self):
        return (self.x, self.y, self.w, self.h)


def _sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c)).upper()


def ler_gt_ocr(caminho: Path) -> list[tuple[int, int, int, int, str]]:
    """Lê o `<id>_gt_ocr.txt` (cp1252). Retorna [(x, y, w, h, texto), ...]."""
    regs = []
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            linhas = caminho.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        return regs
    for ln in linhas:
        if not ln.strip() or ln.lower().startswith("x,"):
            continue
        partes = ln.split(",", 4)
        if len(partes) < 5:
            continue
        try:
            x, y, w, h = (int(p) for p in partes[:4])
        except ValueError:
            continue
        regs.append((x, y, w, h, partes[4].strip()))
    return regs


def _score_ocr(imagem_bgr) -> float:
    """Soma das confianças de OCR de 'palavras' (>=3 letras). Máximo quando a
    imagem está de pé — usado para desambiguar a orientação."""
    try:
        dados = pytesseract.image_to_data(imagem_bgr, output_type=pytesseract.Output.DICT)
    except Exception:
        return 0.0
    score = 0.0
    for txt, conf in zip(dados["text"], dados["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c > 0 and sum(ch.isalpha() for ch in txt) >= 3:
            score += c
    return score


_OPS_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def detectar_rotacao(imagem_bgr, regs, score_min: float = 60.0) -> int | None:
    """Detecta a rotação (graus horários p/ endireitar) de forma determinística:
    o EIXO do texto (horizontal/vertical) vem do aspecto das caixas ground-truth
    do BID (confiável), e a ambiguidade restante (0×180 ou 90×270) é resolvida
    pelo score de OCR nas duas orientações candidatas. Bem mais robusto que o
    tesseract OSD, que dava falsos '0' e virava documentos de cabeça pra baixo.
    Retorna None se o texto for insuficiente para decidir com segurança."""
    horizontais = sum(1 for (x, y, w, h, t) in regs if w >= h)
    candidatos = [0, 180] if horizontais >= (len(regs) - horizontais) else [90, 270]
    melhor, melhor_score = None, -1.0
    for rot in candidatos:
        op = _OPS_ROT[rot]
        img = imagem_bgr if op is None else cv2.rotate(imagem_bgr, op)
        s = _score_ocr(img)
        if s > melhor_score:
            melhor_score, melhor = s, rot
    return melhor if melhor_score >= score_min else None


def _rot_ponto(x, y, w, h, W, H, rot):
    """Mapeia uma caixa (x,y,w,h) para o referencial da imagem rotacionada por
    `rot` graus horários (0/90/180/270). Retorna (x,y,w,h) nova."""
    if rot == 0:
        return x, y, w, h
    cantos = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if rot == 90:      # cv2.ROTATE_90_CLOCKWISE -> novas dims (H, W)
        m = [(H - 1 - py, px) for px, py in cantos]
    elif rot == 180:   # cv2.ROTATE_180
        m = [(W - 1 - px, H - 1 - py) for px, py in cantos]
    elif rot == 270:   # cv2.ROTATE_90_COUNTERCLOCKWISE
        m = [(py, W - 1 - px) for px, py in cantos]
    else:
        return x, y, w, h
    xs = [p[0] for p in m]; ys = [p[1] for p in m]
    nx, ny = min(xs), min(ys)
    return nx, ny, max(xs) - nx, max(ys) - ny


def corrigir_orientacao(imagem_bgr, regs, rot):
    """Rotaciona imagem e caixas para 'de pé' segundo `rot` (graus horários)."""
    if rot == 0:
        return imagem_bgr, regs
    H, W = imagem_bgr.shape[:2]
    op = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
    img2 = cv2.rotate(imagem_bgr, op)
    regs2 = []
    for (x, y, w, h, t) in regs:
        nx, ny, nw, nh = _rot_ponto(x, y, w, h, W, H, rot)
        regs2.append((nx, ny, nw, nh, t))
    return img2, regs2


def _eh_rotulo(texto: str) -> bool:
    toks = _sem_acento(texto).replace(".", " ").replace("/", " ").split()
    return any(t in _STOP for t in toks)


def _nome_like(texto: str) -> bool:
    toks = [t for t in texto.split() if any(c.isalpha() for c in t)]
    return len(toks) >= 2 and all(len(t) >= 2 for t in toks) and not any(c.isdigit() for c in texto)


def _centro(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def classificar_campos(regs) -> list[CampoBID]:
    """Identifica caixas de dados pessoais editáveis (exclui rótulos/cabeçalhos)."""
    rotulos = {}  # keyword -> lista de caixas de rótulo
    for (x, y, w, h, t) in regs:
        norm = _sem_acento(t)
        for chave in ("NOME", "FILIACAO"):
            if chave in norm and w > h:  # rótulo horizontal
                rotulos.setdefault(chave, []).append((x, y, w, h))

    def _rotulo_mais_proximo(caixa):
        melhor, dist = None, 1e18
        for chave, caixas in rotulos.items():
            for r in caixas:
                d = (_centro(caixa)[0] - _centro(r)[0]) ** 2 + (_centro(caixa)[1] - _centro(r)[1]) ** 2
                if d < dist:
                    dist, melhor = d, chave
        return melhor

    campos = []
    for (x, y, w, h, t) in regs:
        if w <= h:            # texto não-horizontal no referencial corrigido: pula
            continue
        if _eh_rotulo(t):     # é rótulo/cabeçalho
            continue
        caixa = (x, y, w, h)
        if RE_DATA.match(t):
            tipo = "data"
        elif RE_CPF.match(t):
            tipo = "cpf"
        elif RE_NUM_LONGO.match(t):
            tipo = "registro"
        elif _nome_like(t):
            tipo = "filiacao" if _rotulo_mais_proximo(caixa) == "FILIACAO" else "nome"
        else:
            continue
        campos.append(CampoBID(tipo, x, y, w, h, t))
    return campos


def carregar_documento(caminho_in: Path):
    """Carrega um doc do BID: imagem corrigida + campos editáveis. Retorna None
    se a imagem/label faltarem ou o OSD não for confiável (orientação incerta)."""
    gt = caminho_in.with_name(caminho_in.name.replace("_in.jpg", "_gt_ocr.txt"))
    if not gt.exists():
        return None
    img = cv2.imread(str(caminho_in))
    if img is None:
        return None
    regs = ler_gt_ocr(gt)
    if not regs:
        return None
    rot = detectar_rotacao(img, regs)
    if rot is None:
        return None
    img2, regs2 = corrigir_orientacao(img, regs, rot)
    campos = classificar_campos(regs2)
    if not campos:
        return None
    return {"imagem": img2, "campos": campos, "rotacao": rot, "regs": regs2}


def amostrar(pasta_bid: Path, classes, por_classe: int, rng) -> list[tuple[Path, str]]:
    """Amostra `por_classe` documentos (_in.jpg) de cada classe. Retorna
    [(caminho_in, tipo_documento_lowercase), ...]."""
    escolhidos = []
    for classe in classes:
        docs = sorted((pasta_bid / classe).glob("*_in.jpg"))
        rng.shuffle(docs)
        for p in docs[:por_classe]:
            escolhidos.append((p, classe.lower()))
    return escolhidos
