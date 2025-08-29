# -*- coding: utf-8 -*-
"""
Python 3.12 — Gera 2 Matrix Bubble Charts a partir de um CSV padronizado
(artigos_selecionados_<Base>.csv), seguindo o mesmo padrão de normalização já usado.

Gráficos:
1) Eixo vertical = "Dados Utilizados"; eixos laterais = "Tipo de Trabalho" (esquerda)
   e "Contribuição do Trabalho" (direita).
2) Eixo vertical = "Técnicas Empregadas"; eixos laterais = "Tipo de Trabalho" (esquerda)
   e "Contribuição do Trabalho" (direita).

Parâmetro:
- SHOW_PERCENT = True/False  -> liga/desliga a exibição das porcentagens nas bolhas.
  (No lado esquerdo, % é relativa ao total da coluna "Tipo de Trabalho";
   no lado direito, % é relativa ao total da coluna "Contribuição do Trabalho".)

Saída:
- PNGs salvos em ./out/<Base>/:
    * fig_matrix_dados_utilizados_<Base>.png
    * fig_matrix_tecnicas_empregadas_<Base>.png

Requisitos:
- pandas, matplotlib
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


# ===================== CONFIG =====================
INPUT_CSV = Path("./ArtigosSelecionados/artigos_selecionados_TodasBases.csv")
OUT_ROOT = Path("./Graficos/MatrixBubbleCharts")
SHOW_PERCENT = False  # << ajuste aqui se quiser sem porcentagem
FIGSIZE = (12, 10)   # tamanho base do gráfico
DPI = 200
# ==================================================


# ------------- Normalização / utilidades -------------
def normalize_for_matching(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_value(s: str) -> str:
    """
    Remove acentos e caracteres especiais dos valores (mantém letras/números/espaço).
    Mantém caixa original.
    """
    if s is None:
        return ""
    s_decomp = unicodedata.normalize("NFD", str(s))
    s_noacc = "".join(ch for ch in s_decomp if not unicodedata.combining(ch))
    s_alnum_space = re.sub(r"[^0-9A-Za-z ]+", " ", s_noacc)
    s_clean = re.sub(r"\s+", " ", s_alnum_space).strip()
    return s_clean


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    norm_map: dict[str, str] = {normalize_for_matching(c): c for c in df.columns}
    wanted_norms = [normalize_for_matching(c) for c in candidates]
    for wn in wanted_norms:
        if wn in norm_map:
            return norm_map[wn]
    for wn in wanted_norms:
        for nc, orig in norm_map.items():
            if wn in nc:
                return orig
    # fallback: melhor por palavras-chave
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


def split_items(cell) -> list[str]:
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;/|]+", str(cell))
    return [p.strip() for p in parts if p and p.strip()]


def infer_base_name(csv_path: Path) -> str:
    stem = csv_path.stem
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", stem)
    tokens = [normalize_value(t) for t in tokens]
    # tenta achar "TodasBases"
    for t in tokens:
        if t.lower() == "todasbases":
            return "TodasBases"
    # última palavra "significativa"
    generic = {"artigos", "selecionados", "selecionadas", "selecionado", "selecionada", "exportacao", "v", "versao"}
    meaningful = [t for t in tokens if t.lower() not in generic]
    candidate = meaningful[-1] if meaningful else "Base"
    return candidate.replace(" ", "") or "Base"
# -----------------------------------------------------


# ------------- Construção das tabelas cruzadas -------------
def series_to_sets(series: pd.Series) -> list[set[str]]:
    """
    Converte uma série de strings multivalor em uma lista de conjuntos normalizados por linha.
    """
    out: list[set[str]] = []
    for cell in series.tolist():
        items = split_items(cell)
        norm_items = {normalize_value(x) for x in items}
        norm_items.discard("")
        out.append(norm_items)
    return out


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=lambda s: s.casefold())


def cross_count(vert_sets: list[set[str]], side_sets: list[set[str]]) -> dict[tuple[str, str], int]:
    """
    Conta pares (v, s) por linha (um artigo conta no máximo 1 por par).
    """
    counter: Counter[tuple[str, str]] = Counter()
    for vset, sset in zip(vert_sets, side_sets):
        for v in vset:
            for s in sset:
                counter[(v, s)] += 1
    return dict(counter)


def totals_per_side(counter_vs: dict[tuple[str, str], int]) -> dict[str, int]:
    tot: Counter[str] = Counter()
    for (_, s), c in counter_vs.items():
        tot[s] += c
    return dict(tot)


def totals_per_vert(counter_vs: dict[tuple[str, str], int]) -> dict[str, int]:
    tot: Counter[str] = Counter()
    for (v, _), c in counter_vs.items():
        tot[v] += c
    return dict(tot)
# -----------------------------------------------------------


# ------------- Plot: Matrix Bubble Chart (duas colunas laterais) -------------
def plot_matrix_bubble(
    title: str,
    vert_labels: list[str],
    left_labels: list[str],
    right_labels: list[str],
    counts_left: dict[tuple[str, str], int],
    counts_right: dict[tuple[str, str], int],
    show_percent: bool,
    out_path: Path,
    figsize: tuple[int, int] = FIGSIZE,
    dpi: int = DPI,
) -> None:
    """
    Desenha um gráfico com:
      - eixo vertical (categorias verticais em y)
      - colunas à esquerda (left_labels)
      - colunas à direita (right_labels)
      - círculo por par (v, coluna) com tamanho proporcional à contagem
      - opcionalmente mostra porcentagem (por coluna)
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    n_y = len(vert_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    # posições em grade
    ys = list(range(n_y))[::-1]  # topo -> base
    y_pos = {v: y for v, y in zip(vert_labels, ys)}
    x_left = {lbl: -(n_left - i) for i, lbl in enumerate(left_labels)}   # ... -3, -2, -1
    x_right = {lbl: (i + 1) for i, lbl in enumerate(right_labels)}       # 1, 2, 3 ...

    # escala dos círculos
    max_count = 1
    if counts_left:
        max_count = max(max_count, max(counts_left.values()))
    if counts_right:
        max_count = max(max_count, max(counts_right.values()))
    s_min, s_max = 150, 4500

    def size_for(c: int) -> float:
        # não-linear suave
        return s_min if c <= 1 else s_min + (s_max - s_min) * (c - 1) / (max_count - 1 if max_count > 1 else 1)

    # grid e linha central
    all_x = list(x_left.values()) + [0] + list(x_right.values())
    ax.axvline(0, linewidth=1.5, color="black")
    for x in all_x:
        if x == 0:
            continue
        ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
    for y in ys:
        ax.axhline(y, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)

    # textos do eixo vertical no centro
    for v in vert_labels:
        ax.text(0, y_pos[v], v, ha="center", va="center", fontsize=10)

    # rótulos de colunas embaixo
    for lbl, x in x_left.items():
        ax.text(x, -0.8, lbl, ha="center", va="top", fontsize=10)
    for lbl, x in x_right.items():
        ax.text(x, -0.8, lbl, ha="center", va="top", fontsize=10)

    # totais por coluna para calcular %
    tot_left = totals_per_side(counts_left)
    tot_right = totals_per_side(counts_right)

    # desenha LEFT
    for (v, s), c in counts_left.items():
        if c <= 0:
            continue
        ax.scatter(
            [x_left[s]], [y_pos[v]],
            s=size_for(c),
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
        )
        # contagem dentro
        ax.text(x_left[s], y_pos[v], f"{c}", ha="center", va="center", fontsize=10)
        # % opcional (por coluna)
        if show_percent and tot_left.get(s, 0) > 0:
            pct = 100.0 * c / tot_left[s]
            ax.text(x_left[s], y_pos[v] + 0.28, f"{pct:.2f}%", ha="center", va="bottom", fontsize=7)

    # desenha RIGHT
    for (v, s), c in counts_right.items():
        if c <= 0:
            continue
        ax.scatter(
            [x_right[s]], [y_pos[v]],
            s=size_for(c),
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
        )
        ax.text(x_right[s], y_pos[v], f"{c}", ha="center", va="center", fontsize=10)
        if show_percent and tot_right.get(s, 0) > 0:
            pct = 100.0 * c / tot_right[s]
            ax.text(x_right[s], y_pos[v] + 0.28, f"{pct:.2f}%", ha="center", va="bottom", fontsize=7)

    # título e ajustes
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlim(min(all_x) - 0.8, max(all_x) + 0.8)
    ax.set_ylim(-1.2, max(ys) + 1)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
# -----------------------------------------------------------------------------


# ------------- Pipeline principal -------------
def build_and_plot(csv_path: Path, show_percent: bool = SHOW_PERCENT) -> None:
    df = pd.read_csv(csv_path)  # mesma codificação usada antes (CSV padronizado)
    base = infer_base_name(csv_path)
    out_dir = OUT_ROOT / base
    out_dir.mkdir(parents=True, exist_ok=True)

    # Descobre colunas
    col_dados = find_column(df, ["Dados Utilizados", "Dados utilizados", "Dados_Utilizados"])
    col_tecs = find_column(df, ["Técnicas Empregadas", "Tecnicas Empregadas", "Tecnicas", "Técnicas"])
    col_tipo = find_column(df, ["Tipo de Trabalho", "Tipo do Trabalho", "Tipo Trabalho", "Tipo"])
    col_contrib = find_column(df, ["Contribuição do Trabalho", "Contribuicao do Trabalho", "Contribuição de Trabalho", "Contribuicao de Trabalho"])

    # Converte para conjuntos por linha
    dados_sets = series_to_sets(df[col_dados])
    tecs_sets = series_to_sets(df[col_tecs])
    tipo_sets = series_to_sets(df[col_tipo])
    contrib_sets = series_to_sets(df[col_contrib])

    # Rótulos ordenados (por frequência decrescente do eixo vertical)
    def sort_by_freq(all_sets: list[set[str]]) -> list[str]:
        c = Counter()
        for s in all_sets:
            c.update(s)
        return [k for k, _ in c.most_common()]

    vert_dados = sort_by_freq(dados_sets)
    vert_tecs = sort_by_freq(tecs_sets)
    left_labels = sort_by_freq(tipo_sets)
    right_labels = sort_by_freq(contrib_sets)

    # Contagens cruzadas
    counts_dados_left = cross_count(dados_sets, tipo_sets)
    counts_dados_right = cross_count(dados_sets, contrib_sets)
    counts_tecs_left = cross_count(tecs_sets, tipo_sets)
    counts_tecs_right = cross_count(tecs_sets, contrib_sets)

    # Plot 1: Dados Utilizados
    plot_matrix_bubble(
        title="Dados Utilizados",
        vert_labels=vert_dados,
        left_labels=left_labels,
        right_labels=right_labels,
        counts_left={(v, s): c for (v, s), c in counts_dados_left.items()},
        counts_right={(v, s): c for (v, s), c in counts_dados_right.items()},
        show_percent=show_percent,
        out_path=out_dir / f"fig_matrix_dados_utilizados_{base}.png",
    )

    # Plot 2: Técnicas Empregadas
    plot_matrix_bubble(
        title="Técnicas Empregadas",
        vert_labels=vert_tecs,
        left_labels=left_labels,
        right_labels=right_labels,
        counts_left={(v, s): c for (v, s), c in counts_tecs_left.items()},
        counts_right={(v, s): c for (v, s), c in counts_tecs_right.items()},
        show_percent=show_percent,
        out_path=out_dir / f"fig_matrix_tecnicas_empregadas_{base}.png",
    )


if __name__ == "__main__":
    build_and_plot(INPUT_CSV, show_percent=SHOW_PERCENT)
