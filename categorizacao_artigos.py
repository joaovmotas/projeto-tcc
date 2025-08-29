# -*- coding: utf-8 -*-
"""
categorizacao_artigos.py — Python 3.12

NOVA LÓGICA: a entrada agora é um ÚNICO CSV contendo artigos de TODAS as bases.
Existe uma coluna "Base" indicando a qual base cada artigo pertence.

O script segmenta por "Base" e gera as MESMAS tabelas para cada base com sufixo
do nome da base. Além disso, gera as tabelas agregadas (com todos os artigos)
em out/TodasBases/ sem sufixo.

Tabelas geradas:
- contribuicao_trabalho_<BASE>.csv
- dados_utilizados_<BASE>.csv
- dados_utilizados_combinados_<BASE>.csv
- tecnicas_empregadas_<BASE>.csv
- tipo_trabalho_<BASE>.csv
e em out/TodasBases/:
- contribuica_trabalho.csv
- dados_utilizados.csv
- dados_utilizados_combinados.csv
- tecnicas_empregadas.csv
- tipo_trabalho.csv

Tratamento de erros:
- Arquivo inexistente -> saída 1
- Formato inválido (colunas obrigatórias ausentes) -> saída 2
"""

from __future__ import annotations

import sys
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


# ========== CONFIGURAÇÃO ==========
INPUT_FILE = Path("./ArtigosSelecionados/artigos_selecionados_TodasBases.csv")  # caminho do CSV único
OUT_ROOT = Path("./ArtigosCategorizados")                           # pasta raiz de saída
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
def split_items(cell) -> list[str]:
    """
    Divide uma célula em itens usando separadores comuns.
    NÃO divide por ' e ' para não quebrar nomes compostos.
    *IMPORTANTE*: NÃO converte previamente para string; se for NaN, retorna [].
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
    for cell in series.tolist():   # preserva NaN como NaN
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
    for cell in series.tolist():   # preserva NaN como NaN
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


# ---------- Sanitização do valor da base ----------
def sanitize_base_value(val: str) -> str:
    """
    Converte o valor da coluna 'Base' em um identificador de pasta/sufixo:
    - remove acentos e caracteres especiais
    - tira espaços
    - se ficar vazio, usa 'Base'
    """
    name = normalize_strict_value(str(val)).replace(" ", "")
    return name or "Base"


# ---------- Processamento de um DataFrame (slice por base) ----------
def process_df_slice(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Recebe um DataFrame já filtrado (uma base específica ou o conjunto completo)
    e retorna os 5 DataFrames de saída.
    """
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
    col_tipo_trab = find_column(
        df,
        [
            "Tipo de Trabalho",
            "Tipo do Trabalho",
            "Tipo Trabalho",
            "Tipo",
        ],
    )

    cnt_dados = count_by_classification_counter(df[col_dados])
    cnt_tec = count_by_classification_counter(df[col_tec])
    cnt_contrib = count_by_classification_counter(df[col_contrib])
    cnt_dados_combo = count_by_combination_counter(df[col_dados])
    cnt_tipo_trab = count_by_classification_counter(df[col_tipo_trab])

    return {
        "dados_utilizados": counter_to_df(cnt_dados),
        "tecnicas_empregadas": counter_to_df(cnt_tec),
        "contribuica_trabalho": counter_to_df(cnt_contrib),
        "dados_utilizados_combinados": counter_to_df(cnt_dados_combo),
        "tipo_trabalho": counter_to_df(cnt_tipo_trab),
    }


# ---------- Escrita ----------
def write_per_base_outputs(base_name: str, tables: dict[str, pd.DataFrame]) -> None:
    """
    Escreve os CSVs da base nos nomes/pastas solicitados.
    """
    base_dir = OUT_ROOT / base_name
    ensure_dir(base_dir)

    tables["contribuica_trabalho"].to_csv(base_dir / f"contribuicao_trabalho_{base_name}.csv", index=False, encoding="utf-8")
    tables["dados_utilizados"].to_csv(base_dir / f"dados_utilizados_{base_name}.csv", index=False, encoding="utf-8")
    tables["dados_utilizados_combinados"].to_csv(base_dir / f"dados_utilizados_combinados_{base_name}.csv", index=False, encoding="utf-8")
    tables["tecnicas_empregadas"].to_csv(base_dir / f"tecnicas_empregadas_{base_name}.csv", index=False, encoding="utf-8")
    tables["tipo_trabalho"].to_csv(base_dir / f"tipo_trabalho_{base_name}.csv", index=False, encoding="utf-8")


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
    agg_tables["tipo_trabalho"].to_csv(tgt / "tipo_trabalho.csv", index=False, encoding="utf-8")


# ---------- Validação do formato (inclui coluna 'Base') ----------
REQUIRED_BLOCKS = {
    "dados": ["Dados Utilizados", "Dados utilizados", "Dados_Utilizados"],
    "tecnicas": ["Técnicas Empregadas", "Tecnicas Empregadas", "Tecnicas", "Técnicas"],
    "contrib": ["Contribuição do Trabalho", "Contribuicao do Trabalho", "Contribuição de Trabalho", "Contribuicao de Trabalho"],
    "tipo": ["Tipo de Trabalho", "Tipo do Trabalho", "Tipo Trabalho", "Tipo"],
    "base": ["Base", "Fonte", "Base de Dados", "Fonte de Dados"],
}


def validate_input_format(df: pd.DataFrame) -> dict[str, str]:
    """
    Verifica a presença das colunas conceituais exigidas, incluindo 'Base'.
    Retorna um mapeamento chave interna -> nome real da coluna.
    """
    resolved: dict[str, str] = {}
    missing = []
    for key, variants in REQUIRED_BLOCKS.items():
        try:
            resolved[key] = find_column(df, variants)
        except KeyError:
            missing.append((key, variants))

    if missing:
        msg = ["[ERRO] Formato inválido: não foi possível localizar as seguintes colunas obrigatórias:"]
        for key, variants in missing:
            msg.append(f"  - {key}: aceita variações {variants}")
        msg.append(f"Colunas disponíveis no arquivo: {list(df.columns)}")
        raise ValueError("\n".join(msg))

    return resolved


# ---------- MAIN ----------
def main():
    ensure_dir(OUT_ROOT)

    # 1) Arquivo existe?
    if not INPUT_FILE.exists():
        print(f"[ERRO] Arquivo não encontrado: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    # 2) Leitura segura
    try:
        df = read_csv_safely(INPUT_FILE)
    except Exception as e:
        print(f"[ERRO] Falha ao ler o CSV '{INPUT_FILE}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Validação do formato (inclui coluna 'Base')
    try:
        cols = validate_input_format(df)
    except ValueError as ve:
        print(str(ve), file=sys.stderr)
        print("[DICA] Garanta que o arquivo segue o mesmo padrão do 'artigos_selecionados_TodasBases.csv'.", file=sys.stderr)
        sys.exit(2)

    base_col = cols["base"]

    # 4) Tabelas agregadas (TodasBases)
    print("\n[PROCESSANDO] Agregado (todas as bases)")
    agg_tables = process_df_slice(df)
    write_aggregated_outputs(agg_tables)

    # 5) Tabelas por base (segmentando pela coluna 'Base')
    bases_raw = (
        df[base_col]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    if not bases_raw:
        print("[AVISO] Coluna 'Base' não possui valores utilizáveis. Nada será gerado por base.", file=sys.stderr)
    else:
        print(f"[INFO] Bases detectadas: {bases_raw}")

    for raw in sorted(bases_raw, key=lambda s: s.casefold()):
        base_name = sanitize_base_value(raw)
        print(f"  - Gerando tabelas para base: {raw}  -> pasta/sufixo: {base_name}")
        df_base = df[df[base_col].astype(str).str.strip() == raw]
        tables = process_df_slice(df_base)
        write_per_base_outputs(base_name, tables)

    # 6) Prints rápidos do agregado
    print("\n[OK] Arquivos agregados gerados em ./ArtigosCategorizados/TodasBases/")
    print("[OK] Arquivos por base gerados em ./ArtigosCategorizados/<Base>/")
    print("\n[RESUMO — TodasBases]")
    for k, dfx in agg_tables.items():
        print(f"\n=== {k} ===")
        print(dfx.to_string(index=False))


if __name__ == "__main__":
    main()
