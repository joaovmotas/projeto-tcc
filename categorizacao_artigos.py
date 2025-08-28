# -*- coding: utf-8 -*-
"""
Python 3.12 — Processa múltiplos CSVs de bases (ex.: artigos_selecionados_IEEE.csv, artigos_selecionados_SpringerLink.csv)
e gera saídas por base e agregadas (TodasBases).

Saída por base (dentro de out/<NOMEDABASE>/):
- contribuicao_trabalho_<NOMEDABASE>.csv
- dados_utilizados_<NOMEDABASE>.csv
- dados_utilizados_combinados_<NOMEDABASE>.csv
- tecnicas_empregadas_<NOMEDABASE>.csv
(REMOVIDO: não gera mais "dados_utilizados.csv" sem sufixo por base)

Saída agregada (dentro de out/TodasBases/):
- contribuica_trabalho.csv
- dados_utilizados.csv
- dados_utilizados_combinados.csv
- tecnicas_empregadas.csv
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


# ========== CONFIGURAÇÃO ==========
INPUT_DIR = Path("./artigos_selecionados")     # Raiz onde estão os CSVs de entrada
DISCOVER_PATTERNS = ["*.csv"]     # Padrões de descoberta
OUT_ROOT = Path("./out")          # Pasta de saída
# ==================================


# ---------- Utilidades de IO ----------
def read_csv_safely(csv_path: str | Path) -> pd.DataFrame:
    """Lê CSV tentando com utf-8 e caindo para latin-1 se necessário."""
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- Normalizações ----------
def normalize_for_matching(s: str) -> str:
    """
    Normaliza para matching de nomes de colunas:
    - remove acentos
    - baixa caixa
    - troca não alfanum por espaço
    - comprime espaços
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_strict_value(s: str) -> str:
    """
    Normaliza valores das classificações:
    - remove acentos
    - remove caracteres especiais (mantém letras, números e espaço)
    - comprime espaços
    - preserva maiúsculas/minúsculas originais
    """
    if s is None:
        return ""
    s_decomp = unicodedata.normalize("NFD", str(s))
    s_noacc = "".join(ch for ch in s_decomp if not unicodedata.combining(ch))
    s_alnum_space = re.sub(r"[^0-9A-Za-z ]+", " ", s_noacc)
    s_clean = re.sub(r"\s+", " ", s_alnum_space).strip()
    return s_clean


# ---------- Descoberta de colunas ----------
def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Encontra uma coluna no DataFrame tentando equivalências por normalização e também substring.
    Lança KeyError com mensagem amigável se não encontrar.
    """
    norm_map: dict[str, str] = {normalize_for_matching(c): c for c in df.columns}
    wanted_norms = [normalize_for_matching(c) for c in candidates]

    # 1) match exato normalizado
    for wn in wanted_norms:
        if wn in norm_map:
            return norm_map[wn]

    # 2) substring
    for wn in wanted_norms:
        for norm_col, original in norm_map.items():
            if wn in norm_col:
                return original

    # 3) melhor esforço por palavras-chave
    keywords = set(" ".join(wanted_norms).split())
    best = None
    best_score = 0
    for norm_col, original in norm_map.items():
        score = sum(1 for k in keywords if k and k in norm_col)
        if score > best_score:
            best = original
            best_score = score
    if best and best_score > 0:
        return best

    available = ", ".join(map(str, df.columns))
    wanted = " | ".join(candidates)
    raise KeyError(
        f"Não encontrei coluna para: {wanted}\n"
        f"Colunas disponíveis: {available}"
    )


# ---------- Parsing de itens ----------
def split_items(cell: str) -> list[str]:
    """
    Divide uma célula em itens usando separadores comuns.
    NÃO divide por ' e ' para não quebrar nomes compostos.
    """
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;/|]+", str(cell))
    return [p.strip() for p in parts if p and p.strip()]


# ---------- Contagens ----------
def count_by_classification_counter(series: pd.Series) -> Counter:
    """
    Retorna um Counter de classes (cada artigo contribui no máximo 1 vez por classe).
    """
    counter = Counter()
    for cell in series.astype(str).tolist():
        items = split_items(cell)
        norm_items = {normalize_strict_value(x) for x in items}
        norm_items.discard("")  # remove vazios
        counter.update(norm_items)
    return counter


def build_combination_label(items: set[str]) -> str:
    """
    Constrói um rótulo determinístico para a COMBINAÇÃO de itens:
    - Ordena alfabeticamente sem considerar caixa
    - Junta com ' | ' e aplica normalização estrita
    """
    if not items:
        return ""
    ordered = sorted(items, key=lambda s: s.casefold())
    raw = " | ".join(ordered)
    return normalize_strict_value(raw)


def count_by_combination_counter(series: pd.Series) -> Counter:
    """
    Retorna um Counter onde cada artigo contribui para exatamente 1 combinação (conjunto) dos seus itens.
    """
    counter = Counter()
    for cell in series.astype(str).tolist():
        items = split_items(cell)
        norm_items = {normalize_strict_value(x) for x in items}
        norm_items.discard("")
        combo_label = build_combination_label(norm_items)
        if combo_label:
            counter.update([combo_label])
    return counter


def counter_to_df(counter: Counter) -> pd.DataFrame:
    """Converte Counter em DataFrame ordenado no formato final."""
    df_out = (
        pd.DataFrame(
            {"Classificação": list(counter.keys()), "Número de Artigos": list(counter.values())}
        )
        .sort_values(["Número de Artigos", "Classificação"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df_out


# ---------- Inferência do nome da base a partir do arquivo ----------
KNOWN_BASES = [
    "SpringerLink", "IEEE", "ACM", "Scopus", "ScienceDirect", "Web of Science", "WoS",
    "SciELO", "PubMed", "arXiv", "Google Scholar", "GoogleScholar", "ERIC", "Elsevier"
]


def infer_base_name(csv_path: Path) -> str:
    """
    Infere um nome de base a partir do nome do arquivo.
    1) Procura KNOWN_BASES como substring (case-insensitive)
    2) Caso não encontre, usa última palavra alfabética "significativa"
    3) Normaliza para [A-Za-z0-9_], sem espaços
    """
    stem = csv_path.stem
    low = stem.lower()
    for base in KNOWN_BASES:
        if base.lower().replace(" ", "") in low.replace(" ", ""):
            candidate = base
            break
    else:
        tokens = re.findall(r"[A-Za-zÀ-ÿ]+", stem)
        generic = {t.lower() for t in ["artigos", "selecionados", "selecionadas", "exportacao", "bases", "base", "selecionado", "selecionada", "v", "versao"]}
        meaningful = [t for t in tokens if t.lower() not in generic]
        candidate = meaningful[-1] if meaningful else "Base"

    candidate = normalize_strict_value(candidate).replace(" ", "")
    return candidate or "Base"


# ---------- Processamento de um arquivo ----------
def process_single_csv(csv_path: Path) -> dict[str, pd.DataFrame]:
    """
    Processa um CSV e retorna os 4 DataFrames:
    - dados_utilizados
    - tecnicas_empregadas
    - contribuica_trabalho
    - dados_utilizados_combinados
    """
    df = read_csv_safely(csv_path)

    col_dados = find_column(df, ["Dados Utilizados", "Dados utilizados", "Dados_Utilizados"])
    col_tec = find_column(df, ["Técnicas Empregadas", "Tecnicas Empregadas", "Tecnicas", "Técnicas"])
    col_contrib = find_column(
        df,
        [
            "Contribuição de Trabalho",
            "Contribuicao de Trabalho",
            "Contribuição do Trabalho",
            "Contribuicao do Trabalho",
            "Contribuicao Trabalho",
        ],
    )

    cnt_dados = count_by_classification_counter(df[col_dados])
    cnt_tec = count_by_classification_counter(df[col_tec])
    cnt_contrib = count_by_classification_counter(df[col_contrib])
    cnt_dados_combo = count_by_combination_counter(df[col_dados])

    return {
        "dados_utilizados": counter_to_df(cnt_dados),
        "tecnicas_empregadas": counter_to_df(cnt_tec),
        "contribuica_trabalho": counter_to_df(cnt_contrib),
        "dados_utilizados_combinados": counter_to_df(cnt_dados_combo),
    }


def write_per_base_outputs(base_name: str, tables: dict[str, pd.DataFrame]) -> None:
    """
    Escreve os CSVs da base nos nomes/pastas solicitados.
    (Sem gerar "dados_utilizados.csv" sem sufixo.)
    """
    base_dir = OUT_ROOT / base_name
    ensure_dir(base_dir)

    tables["contribuica_trabalho"].to_csv(base_dir / f"contribuicao_trabalho_{base_name}.csv", index=False, encoding="utf-8")
    tables["dados_utilizados"].to_csv(base_dir / f"dados_utilizados_{base_name}.csv", index=False, encoding="utf-8")
    tables["dados_utilizados_combinados"].to_csv(base_dir / f"dados_utilizados_combinados_{base_name}.csv", index=False, encoding="utf-8")
    tables["tecnicas_empregadas"].to_csv(base_dir / f"tecnicas_empregadas_{base_name}.csv", index=False, encoding="utf-8")
    # REMOVIDO: não escrever "base_dir / 'dados_utilizados.csv'"


def write_aggregated_outputs(agg_tables: dict[str, pd.DataFrame]) -> None:
    """
    Escreve os CSVs agregados (TodasBases) sem sufixo.
    """
    tgt = OUT_ROOT / "TodasBases"
    ensure_dir(tgt)

    agg_tables["contribuica_trabalho"].to_csv(tgt / "contribuica_trabalho.csv", index=False, encoding="utf-8")
    agg_tables["dados_utilizados"].to_csv(tgt / "dados_utilizados.csv", index=False, encoding="utf-8")
    agg_tables["dados_utilizados_combinados"].to_csv(tgt / "dados_utilizados_combinados.csv", index=False, encoding="utf-8")
    agg_tables["tecnicas_empregadas"].to_csv(tgt / "tecnicas_empregadas.csv", index=False, encoding="utf-8")


# ---------- Descoberta de entradas ----------
def discover_input_csvs() -> list[Path]:
    """
    Descobre CSVs no INPUT_DIR conforme padrões, filtrando por nomes que contenham 'selecionad' (selecionados/selecionadas).
    """
    results: list[Path] = []
    for pattern in DISCOVER_PATTERNS:
        results.extend(INPUT_DIR.glob(pattern))
    selected: list[Path] = []
    for p in results:
        stem_norm = normalize_for_matching(p.stem)
        if "selecionad" in stem_norm:
            selected.append(p)
    if not selected:
        selected = [p for p in results if p.is_file() and p.suffix.lower() == ".csv"]
    return sorted(set(selected))


# ---------- Agregação incremental ----------
def sum_counters(c1: Counter, c2: Counter) -> Counter:
    c_out = Counter()
    c_out.update(c1)
    c_out.update(c2)
    return c_out


def main():
    ensure_dir(OUT_ROOT)

    csv_files = discover_input_csvs()
    if not csv_files:
        print(f"[ERRO] Nenhum CSV encontrado em {INPUT_DIR}. Ajuste INPUT_DIR/DISCOVER_PATTERNS.", file=sys.stderr)
        sys.exit(1)

    # Acumuladores globais (para TodasBases)
    agg_cnt_dados = Counter()
    agg_cnt_tec = Counter()
    agg_cnt_contrib = Counter()
    agg_cnt_dados_combo = Counter()

    print(f"[INFO] Encontrados {len(csv_files)} arquivo(s) CSV para processar:")
    for p in csv_files:
        print(f"  - {p}")

    for csv_path in csv_files:
        try:
            base_name = infer_base_name(csv_path)
            print(f"\n[PROCESSANDO] {csv_path.name}  -> base: {base_name}")

            # Processa arquivo
            tables = process_single_csv(csv_path)

            # Grava por base
            write_per_base_outputs(base_name, tables)

            # Atualiza agregados
            def df_to_counter(df: pd.DataFrame) -> Counter:
                return Counter(dict(zip(df["Classificação"], df["Número de Artigos"])))

            agg_cnt_dados = sum_counters(agg_cnt_dados, df_to_counter(tables["dados_utilizados"]))
            agg_cnt_tec = sum_counters(agg_cnt_tec, df_to_counter(tables["tecnicas_empregadas"]))
            agg_cnt_contrib = sum_counters(agg_cnt_contrib, df_to_counter(tables["contribuica_trabalho"]))
            agg_cnt_dados_combo = sum_counters(agg_cnt_dados_combo, df_to_counter(tables["dados_utilizados_combinados"]))

        except Exception as e:
            print(f"[ERRO] Falha ao processar {csv_path}: {e}", file=sys.stderr)

    # Constrói DataFrames agregados
    agg_tables = {
        "dados_utilizados": counter_to_df(agg_cnt_dados),
        "tecnicas_empregadas": counter_to_df(agg_cnt_tec),
        "contribuica_trabalho": counter_to_df(agg_cnt_contrib),
        "dados_utilizados_combinados": counter_to_df(agg_cnt_dados_combo),
    }

    # Grava agregados
    write_aggregated_outputs(agg_tables)

    # Prints rápidos
    print("\n[OK] Arquivos por base gerados em ./out/<Base>/")
    print("[OK] Arquivos agregados gerados em ./out/TodasBases/")
    print("\n[RESUMO — TodasBases]")
    for k, df in agg_tables.items():
        print(f"\n=== {k} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
