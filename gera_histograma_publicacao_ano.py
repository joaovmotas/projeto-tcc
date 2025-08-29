# -*- coding: utf-8 -*-
"""
Python 3.12 — Histograma de quantidade de publicações por ano
Entrada: /mnt/data/artigos_selecionados_TodasBases.csv (mesmo formato usado antes)

O script:
- Lê o CSV com tratamento de encoding.
- Localiza a coluna de ANO (variações aceitas: "Ano", "Ano de Publicação", "Year", etc.).
- Extrai de forma robusta o ano (aceita datas/strings contendo um ano 19xx–20xx).
- Gera um histograma (barras por ano) com grid ao fundo.
- Salva em: ./out/TodasBases/hist_publicacoes_por_ano_TodasBases.png
- Também exporta a tabela de contagem: ./out/TodasBases/publicacoes_por_ano.csv

Tratamento de erros:
- Arquivo inexistente -> saída 1
- Formato inválido (não há coluna de ano reconhecível ou nenhum ano válido) -> saída 2
"""

from __future__ import annotations

import sys
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


# ===================== CONFIG =====================
INPUT_CSV = Path("./ArtigosSelecionados/artigos_selecionados_TodasBases.csv")
OUT_DIR = Path("./Graficos/Histograms/TodasBases")
FIG_NAME = "hist_publicacoes_por_ano_TodasBases.png"
COUNTS_CSV = "publicacoes_por_ano.csv"
FIGSIZE = (12, 6)
DPI = 200
# ==================================================

# Variações aceitas para a coluna de ano (ajuste se necessário)
YEAR_CANDIDATES = [
    "Ano", "Ano de Publicação", "Ano de Publicacao", "Ano Publicação", "Ano Publicacao",
    "Year", "Publication Year", "Ano (Publicação)", "Ano (Publicacao)",
    "Data de Publicação", "Data de Publicacao", "Publication Date", "Date"
]


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
    raise KeyError(f"Não encontrei nenhuma coluna compatível com: {candidates}")


def extract_year(value) -> int | None:
    """
    Extrai um ano (1900–2099) de um valor que pode ser int/float/string/data.
    Retorna None se não conseguir extrair.
    """
    if pd.isna(value):
        return None
    # numérico direto
    if isinstance(value, (int,)):
        if 1900 <= value <= 2099:
            return int(value)
        return None
    if isinstance(value, float):
        iv = int(value)
        if 1900 <= iv <= 2099:
            return iv
    # tenta parse de string
    s = str(value)
    # procura primeiro um padrão de 4 dígitos 19xx ou 20xx
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        year = int(m.group(0))
        if 1900 <= year <= 2099:
            return year
    # tenta converter como data conhecida pelo pandas e pegar o ano
    # tenta converter como data e pegar o ano (pandas >= 2.0)
    try:
        dt = pd.to_datetime(s, errors="raise", dayfirst=False)  # <- REMOVIDO infer_datetime_format
        y = int(dt.year)
        if 1900 <= y <= 2099:
            return y
    except Exception:
        pass
    return None


def count_by_year(series: pd.Series) -> pd.Series:
    """Conta publicações por ano (retorna Series index=ano, values=contagem, ordenada por ano crescente)."""
    years = []
    for v in series.tolist():
        y = extract_year(v)
        if y is not None:
            years.append(y)
    if not years:
        return pd.Series(dtype=int)
    s = pd.Series(years).value_counts().sort_index()
    s.index.name = "Ano"
    s.name = "Número de Publicações"
    return s


def plot_hist_years(counts: pd.Series, out_path: Path, figsize: tuple[int, int] = FIGSIZE, dpi: int = DPI) -> None:
    """Desenha um histograma (barras por ano) com grid ao fundo."""
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    years = counts.index.astype(int).tolist()
    values = counts.values.tolist()

    ax.bar(years, values, zorder=2)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de publicações")
    ax.set_title("Publicações por ano")

    # grid atrás das barras
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=1)

    # limites com pequena margem
    if years:
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)

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

    # 3) Descobrir coluna de ano
    try:
        col_year = find_column(df, YEAR_CANDIDATES)
    except KeyError as ke:
        print(str(ke), file=sys.stderr)
        print("[DICA] O arquivo deve ter uma coluna de ano (ex.: 'Ano', 'Year', 'Ano de Publicação').", file=sys.stderr)
        sys.exit(2)

    # 4) Contar anos
    counts = count_by_year(df[col_year])
    if counts.empty:
        print("[ERRO] Nenhum ano válido encontrado após o parsing. Verifique a coluna de ano.", file=sys.stderr)
        sys.exit(2)

    # 5) Salvar tabela e figura
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT_DIR / COUNTS_CSV, header=True, index=True, encoding="utf-8")
    plot_hist_years(counts, OUT_DIR / FIG_NAME)

    print(f"[OK] Histograma salvo em: {OUT_DIR / FIG_NAME}")
    print(f"[OK] Tabela de contagem salva em: {OUT_DIR / COUNTS_CSV}")


if __name__ == "__main__":
    main()
