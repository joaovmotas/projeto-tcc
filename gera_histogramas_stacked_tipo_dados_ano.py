# -*- coding: utf-8 -*-
"""
Python 3.12 — Histograma (barras empilhadas) dos TIPOS de "Dados Utilizados" por ANO

Entrada: /mnt/data/artigos_selecionados_TodasBases.csv (mesmo formato usado antes)

O script:
- Lê o CSV com tratamento de encoding.
- Localiza colunas de ANO e "Dados Utilizados" (variações aceitas).
- Extrai robustamente o ano (aceita datas/strings contendo 19xx–20xx).
- Divide "Dados Utilizados" em múltiplos itens por artigo, normaliza e deduplica por linha.
- Conta ocorrências por par (ano, tipo_de_dado) e gera:
    * Figura: ./out/TodasBases/hist_dados_utilizados_por_ano_TodasBases.png
    * Tabela pivot (anos x tipos): ./out/TodasBases/dados_utilizados_por_ano.csv

Tratamento de erros:
- Arquivo inexistente -> saída 1
- Formato inválido (colunas obrigatórias ausentes ou nenhum par válido) -> saída 2
"""

from __future__ import annotations

import sys
import re
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===================== CONFIG =====================
INPUT_CSV = Path("./ArtigosSelecionados/artigos_selecionados_TodasBases.csv")
OUT_DIR = Path("./Graficos/Histogramas/TodasBases")
FIG_NAME = "hist_stacked_dados_utilizados_por_ano_TodasBases.png"
COUNTS_CSV = "dados_utilizados_por_ano_stacked.csv"
FIGSIZE = (12, 6)
DPI = 200
# Limite opcional de categorias no gráfico (demais viram "Outros"). Use None para desabilitar.
MAX_CATEGORIES: int | None = 12
# ==================================================

# Variações aceitas para as colunas
YEAR_CANDIDATES = [
    "Ano", "Ano de Publicação", "Ano de Publicacao", "Ano Publicação", "Ano Publicacao",
    "Year", "Publication Year", "Ano (Publicação)", "Ano (Publicacao)",
    "Data de Publicação", "Data de Publicacao", "Publication Date", "Date"
]
DADOS_CANDIDATES = ["Dados Utilizados", "Dados utilizados", "Dados_Utilizados"]


# ---------- Utilidades ----------
def normalize_for_matching(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """Lê CSV tentando UTF-8 e caindo para Latin-1 se necessário."""
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Resolve o nome real da coluna no DataFrame, tolerando acentos e variações."""
    norm_map: dict[str, str] = {normalize_for_matching(c): c for c in df.columns}
    wanted_norms = [normalize_for_matching(c) for c in candidates]
    # match exato
    for wn in wanted_norms:
        if wn in norm_map:
            return norm_map[wn]
    # substring
    for wn in wanted_norms:
        for nc, orig in norm_map.items():
            if wn in nc:
                return orig
    # fallback por palavras-chave
    keywords = set(" ".join(wanted_norms).split())
    best = None
    best_score = 0
    for nc, orig in norm_map.items():
        score = sum(1 for k in keywords if k and k in nc)
        if score > best_score:
            best, best_score = orig, score
    if best and best_score > 0:
        return best
    raise KeyError(f"Não encontrei nenhuma coluna compatível com: {candidates}\nDisponíveis: {list(df.columns)}")


def normalize_value(s: str) -> str:
    """Remove acentos e caracteres especiais dos valores (mantém letras/números/espaço)."""
    if s is None:
        return ""
    s_decomp = unicodedata.normalize("NFD", str(s))
    s_noacc = "".join(ch for ch in s_decomp if not unicodedata.combining(ch))
    s_alnum_space = re.sub(r"[^0-9A-Za-z ]+", " ", s_noacc)
    s_clean = re.sub(r"\s+", " ", s_alnum_space).strip()
    return s_clean


def split_items(cell) -> list[str]:
    """Divide strings multi-valor por vírgula/; / | / quebra de linha. Retorna [] para NaN."""
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;/|]+", str(cell))
    return [p.strip() for p in parts if p and p.strip()]


def extract_year(value) -> int | None:
    """
    Extrai um ano (1900–2099) de um valor que pode ser int/float/string/data.
    Retorna None se não conseguir extrair.
    """
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2099 else None
    if isinstance(value, float):
        iv = int(value)
        return iv if 1900 <= iv <= 2099 else None

    s = str(value)
    # tenta primeiro 19xx|20xx explícito
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        year = int(m.group(0))
        if 1900 <= year <= 2099:
            return year
    # tenta converter como data e pegar o ano (sem infer_datetime_format)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.notna(dt):
        y = int(dt.year)
        if 1900 <= y <= 2099:
            return y
    return None


# ---------- Contagem por (ano, tipo de dado) ----------
def build_counts_by_year_and_data(df: pd.DataFrame, col_year: str, col_dados: str) -> pd.DataFrame:
    """
    Retorna DataFrame pivotado: index=Ano, colunas=tipos de dados, valores=contagem.
    Regras:
      - Cada artigo contribui no máximo 1 por (ano, tipo). (dedup por linha)
      - Valores normalizados sem acentos/caracteres especiais.
    """
    counter: Counter[tuple[int, str]] = Counter()

    for _, row in df.iterrows():
        year = extract_year(row[col_year])
        if year is None:
            continue
        items = split_items(row[col_dados])
        norm_items = {normalize_value(x) for x in items}
        norm_items.discard("")
        if not norm_items:
            continue
        for item in norm_items:
            counter[(year, item)] += 1

    if not counter:
        return pd.DataFrame()

    # converte para DF longo
    records = [{"Ano": y, "Tipo": t, "Contagem": c} for (y, t), c in counter.items()]
    df_long = pd.DataFrame.from_records(records)

    # pivot (anos x tipos)
    pivot = df_long.pivot_table(index="Ano", columns="Tipo", values="Contagem", aggfunc="sum", fill_value=0)
    pivot = pivot.sort_index()
    # ordena colunas por soma total desc
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
    return pivot


# ---------- Plot (barras empilhadas) ----------
def plot_stacked_bars(pivot: pd.DataFrame, out_path: Path, max_categories: int | None = MAX_CATEGORIES,
                      figsize: tuple[int, int] = FIGSIZE, dpi: int = DPI) -> None:
    """
    Desenha barras empilhadas por ano. Se max_categories for definido, mantém
    as N categorias mais frequentes e agrega o restante em 'Outros'.
    (Corrigido: garante np.ndarray[float] para 'vals' e 'bottom'.)
    """
    if pivot.empty:
        raise ValueError("Pivot vazio — nada para plotar.")

    data = pivot.copy()

    if max_categories is not None and data.shape[1] > max_categories:
        top_cols = data.sum(0).sort_values(ascending=False).index[:max_categories].tolist()
        other_cols = [c for c in data.columns if c not in top_cols]
        data_top = data[top_cols].copy()
        # soma das “outras” categorias
        data_top["Outros"] = data[other_cols].sum(axis=1)
        data = data_top

    # Anos como ndarray de int (x do gráfico)
    years = data.index.astype(int).to_numpy()

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # BASE: zeros em float; evita Optional e ExtensionArray
    bottom = np.zeros(len(years), dtype=float)

    # Empilhamento: cada série colunar vira um array float
    for col in data.columns:
        vals = data[col].to_numpy(dtype=float)   # <- garante np.ndarray[float]
        ax.bar(years, vals, bottom=bottom, label=col, zorder=2)
        bottom = bottom + vals                   # <- np.ndarray + np.ndarray

    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de publicações")
    ax.set_title("Tipos de 'Dados Utilizados' por ano (barras empilhadas)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=1)

    if years.size:
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)

    # legenda fora do gráfico se houver muitas categorias
    ncols = 3 if data.shape[1] > 12 else 2 if data.shape[1] > 8 else 1
    ax.legend(loc="upper left", ncol=ncols, fontsize=9, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()



# ---------- MAIN ----------
def main():
    # 1) Arquivo existe?
    if not INPUT_CSV.exists():
        print(f"[ERRO] Arquivo não encontrado: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    # 2) Leitura segura
    try:
        df = read_csv_safely(INPUT_CSV)
    except Exception as e:
        print(f"[ERRO] Falha ao ler o CSV '{INPUT_CSV}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Colunas necessárias
    try:
        col_year = find_column(df, YEAR_CANDIDATES)
        col_dados = find_column(df, DADOS_CANDIDATES)
    except KeyError as ke:
        print(str(ke), file=sys.stderr)
        print("[DICA] O arquivo deve ter colunas equivalentes a 'Ano' e 'Dados Utilizados'.", file=sys.stderr)
        sys.exit(2)

    # 4) Pivot (anos x tipos)
    pivot = build_counts_by_year_and_data(df, col_year, col_dados)
    if pivot.empty:
        print("[ERRO] Nenhum par (ano, tipo de dado) válido encontrado após o parsing.", file=sys.stderr)
        sys.exit(2)

    # 5) Salvar tabela e figura
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUT_DIR / COUNTS_CSV, encoding="utf-8")
    plot_stacked_bars(pivot, OUT_DIR / FIG_NAME, max_categories=MAX_CATEGORIES)

    print(f"[OK] Figura salva em: {OUT_DIR / FIG_NAME}")
    print(f"[OK] Tabela pivot salva em: {OUT_DIR / COUNTS_CSV}")


if __name__ == "__main__":
    main()
