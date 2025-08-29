# -*- coding: utf-8 -*-
"""
Python 3.12 — Matrix Bubble Charts com tratamento de erros.

Gera 2 gráficos:
1) Vertical = "Dados Utilizados"; laterais = "Tipo de Trabalho" (esq.) e "Contribuição do Trabalho" (dir.)
2) Vertical = "Técnicas Empregadas"; laterais = "Tipo de Trabalho" (esq.) e "Contribuição do Trabalho" (dir.)

Tratamento de erros:
- Arquivo de entrada inexistente -> mensagem clara e saída com código 1.
- Formato inválido (não contém as colunas esperadas no padrão do CSV de referência) -> mensagem clara e saída com código 2.

Saída:
- ./out/<Base>/fig_matrix_dados_utilizados_<Base>.png
- ./out/<Base>/fig_matrix_tecnicas_empregadas_<Base>.png
"""

from __future__ import annotations

import sys
import re
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ===================== CONFIG =====================
INPUT_CSV = Path("./ArtigosSelecionados/artigos_selecionados_TodasBases.csv")
OUT_ROOT = Path("./Graficos/MatrixBubbleCharts")
SHOW_PERCENT = True          # True = exibe porcentagens nas bolhas
FIGSIZE = (12, 10)
DPI = 200
# ==================================================

# Colunas esperadas (variações aceitas para compatibilidade)
REQUIRED_COLS = {
    "dados": [
        "Dados Utilizados", "Dados utilizados", "Dados_Utilizados",
    ],
    "tecnicas": [
        "Técnicas Empregadas", "Tecnicas Empregadas", "Tecnicas", "Técnicas",
    ],
    "tipo": [
        "Tipo de Trabalho", "Tipo do Trabalho", "Tipo Trabalho", "Tipo",
    ],
    "contrib": [
        "Contribuição do Trabalho", "Contribuicao do Trabalho",
        "Contribuição de Trabalho", "Contribuicao de Trabalho",
    ],
}


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
    """Remove acentos e caracteres especiais dos valores (mantém letras/números/espaço)."""
    if s is None:
        return ""
    s_decomp = unicodedata.normalize("NFD", str(s))
    s_noacc = "".join(ch for ch in s_decomp if not unicodedata.combining(ch))
    s_alnum_space = re.sub(r"[^0-9A-Za-z ]+", " ", s_noacc)
    s_clean = re.sub(r"\s+", " ", s_alnum_space).strip()
    return s_clean


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


def validate_input_format(df: pd.DataFrame) -> dict[str, str]:
    """
    Valida se o CSV segue o formato esperado (mesmas colunas conceituais do arquivo de referência).
    Retorna um dict com mapeamento das chaves internas -> nomes reais das colunas.
    Lança ValueError em caso de incompatibilidade.
    """
    resolved: dict[str, str] = {}
    missing = []
    for key, variants in REQUIRED_COLS.items():
        try:
            resolved[key] = find_column(df, variants)
        except KeyError:
            missing.append((key, variants))

    if missing:
        msg_lines = [
            "[ERRO] Formato inválido: não foi possível localizar as seguintes colunas obrigatórias:",
        ]
        for key, variants in missing:
            msg_lines.append(f"  - {key}: aceita variações {variants}")
        msg_lines.append(f"Colunas disponíveis no arquivo: {list(df.columns)}")
        raise ValueError("\n".join(msg_lines))

    return resolved


def split_items(cell) -> list[str]:
    """Divide strings multi-valor por vírgula/; / | / quebra de linha. Retorna [] para NaN."""
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;/|]+", str(cell))
    return [p.strip() for p in parts if p and p.strip()]


def series_to_sets(series: pd.Series) -> list[set[str]]:
    """Converte série multi-valor em lista de conjuntos normalizados por linha."""
    out: list[set[str]] = []
    for cell in series.tolist():
        items = split_items(cell)
        norm_items = {normalize_value(x) for x in items}
        norm_items.discard("")
        out.append(norm_items)
    return out


def unique_sorted(values: list[str]) -> list[str]:
    return sorted(set(values), key=lambda s: s.casefold())


def cross_count(vert_sets: list[set[str]], side_sets: list[set[str]]) -> dict[tuple[str, str], int]:
    """Conta pares (v, s) por linha (um artigo conta no máximo 1 por par)."""
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


def infer_base_name(csv_path: Path) -> str:
    stem = csv_path.stem
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", stem)
    tokens = [normalize_value(t) for t in tokens]
    for t in tokens:
        if t.lower() == "todasbases":
            return "TodasBases"
    generic = {"artigos", "selecionados", "selecionadas", "selecionado", "selecionada", "exportacao", "v", "versao"}
    meaningful = [t for t in tokens if t.lower() not in generic]
    candidate = meaningful[-1] if meaningful else "Base"
    return candidate.replace(" ", "") or "Base"


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
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    n_y = len(vert_labels)
    ys = list(range(n_y))[::-1]
    y_pos = {v: y for v, y in zip(vert_labels, ys)}
    x_left = {lbl: -(len(left_labels) - i) for i, lbl in enumerate(left_labels)}
    x_right = {lbl: (i + 1) for i, lbl in enumerate(right_labels)}

    # escala
    max_count = max([1] + list(counts_left.values()) + list(counts_right.values()))
    s_min, s_max = 150, 4500
    def size_for(c: int) -> float:
        return s_min if c <= 1 else s_min + (s_max - s_min) * (c - 1) / (max_count - 1 if max_count > 1 else 1)

    # grade
    all_x = list(x_left.values()) + [0] + list(x_right.values())
    ax.axvline(0, linewidth=1.5, color="black")
    for x in all_x:
        if x != 0:
            ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
    for y in ys:
        ax.axhline(y, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)

    # labels verticais (centro)
    for v in vert_labels:
        ax.text(0, y_pos[v], v, ha="center", va="center", fontsize=10)

    # rótulos de colunas
    for lbl, x in x_left.items():
        ax.text(x, -0.8, lbl, ha="center", va="top", fontsize=10)
    for lbl, x in x_right.items():
        ax.text(x, -0.8, lbl, ha="center", va="top", fontsize=10)

    tot_left = totals_per_side(counts_left)
    tot_right = totals_per_side(counts_right)

    # esquerda
    for (v, s), c in counts_left.items():
        if c <= 0: 
            continue
        ax.scatter([x_left[s]], [y_pos[v]], s=size_for(c), facecolors="none", edgecolors="black", linewidths=1.8)
        ax.text(x_left[s], y_pos[v], f"{c}", ha="center", va="center", fontsize=10)
        if show_percent and tot_left.get(s, 0) > 0:
            pct = 100.0 * c / tot_left[s]
            ax.text(x_left[s], y_pos[v] + 0.28, f"{pct:.2f}%", ha="center", va="bottom", fontsize=7)

    # direita
    for (v, s), c in counts_right.items():
        if c <= 0: 
            continue
        ax.scatter([x_right[s]], [y_pos[v]], s=size_for(c), facecolors="none", edgecolors="black", linewidths=1.8)
        ax.text(x_right[s], y_pos[v], f"{c}", ha="center", va="center", fontsize=10)
        if show_percent and tot_right.get(s, 0) > 0:
            pct = 100.0 * c / tot_right[s]
            ax.text(x_right[s], y_pos[v] + 0.28, f"{pct:.2f}%", ha="center", va="bottom", fontsize=7)

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlim(min(all_x) - 0.8, max(all_x) + 0.8)
    ax.set_ylim(-1.2, max(ys) + 1)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ------------- Pipeline principal -------------
def build_and_plot(csv_path: Path, show_percent: bool = SHOW_PERCENT) -> None:
    # 1) Arquivo existe?
    if not csv_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 2) Leitura segura
    try:
        df = read_csv_safely(csv_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler o CSV '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Validação de formato
    try:
        cols = validate_input_format(df)
    except ValueError as ve:
        print(str(ve), file=sys.stderr)
        print("[DICA] Verifique se o arquivo segue o mesmo padrão de cabeçalhos do CSV de referência.", file=sys.stderr)
        sys.exit(2)

    # 4) Construção de conjuntos por linha
    dados_sets   = series_to_sets(df[cols["dados"]])
    tecs_sets    = series_to_sets(df[cols["tecnicas"]])
    tipo_sets    = series_to_sets(df[cols["tipo"]])
    contrib_sets = series_to_sets(df[cols["contrib"]])

    # 5) Rótulos ordenados por frequência
    def sort_by_freq(all_sets: list[set[str]]) -> list[str]:
        c = Counter()
        for s in all_sets:
            c.update(s)
        return [k for k, _ in c.most_common()]

    vert_dados = sort_by_freq(dados_sets)
    vert_tecs  = sort_by_freq(tecs_sets)
    left_labels  = sort_by_freq(tipo_sets)
    right_labels = sort_by_freq(contrib_sets)

    # Se alguma dimensão estiver vazia, é formato válido mas sem dados -> erro claro
    if not vert_dados and not vert_tecs:
        print("[ERRO] As colunas de eixo vertical estão vazias após o parsing. Verifique os dados.", file=sys.stderr)
        sys.exit(2)
    if not left_labels:
        print("[ERRO] A coluna 'Tipo de Trabalho' não contém valores utilizáveis.", file=sys.stderr)
        sys.exit(2)
    if not right_labels:
        print("[ERRO] A coluna 'Contribuição do Trabalho' não contém valores utilizáveis.", file=sys.stderr)
        sys.exit(2)

    # 6) Contagens cruzadas
    def cross_count(vert_sets: list[set[str]], side_sets: list[set[str]]) -> dict[tuple[str, str], int]:
        counter: Counter[tuple[str, str]] = Counter()
        for vset, sset in zip(vert_sets, side_sets):
            for v in vset:
                for s in sset:
                    counter[(v, s)] += 1
        return dict(counter)

    counts_dados_left   = cross_count(dados_sets, tipo_sets)
    counts_dados_right  = cross_count(dados_sets, contrib_sets)
    counts_tecs_left    = cross_count(tecs_sets,  tipo_sets)
    counts_tecs_right   = cross_count(tecs_sets,  contrib_sets)

    # 7) Plot
    base = infer_base_name(csv_path)
    out_dir = OUT_ROOT / base

    plot_matrix_bubble(
        title="Dados Utilizados",
        vert_labels=vert_dados,
        left_labels=left_labels,
        right_labels=right_labels,
        counts_left=counts_dados_left,
        counts_right=counts_dados_right,
        show_percent=show_percent,
        out_path=out_dir / f"fig_matrix_dados_utilizados_{base}.png",
    )

    plot_matrix_bubble(
        title="Técnicas Empregadas",
        vert_labels=vert_tecs,
        left_labels=left_labels,
        right_labels=right_labels,
        counts_left=counts_tecs_left,
        counts_right=counts_tecs_right,
        show_percent=show_percent,
        out_path=out_dir / f"fig_matrix_tecnicas_empregadas_{base}.png",
    )

    print(f"[OK] Gráficos gerados em: {out_dir}")


if __name__ == "__main__":
    build_and_plot(INPUT_CSV, show_percent=SHOW_PERCENT)
