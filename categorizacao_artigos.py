# -*- coding: utf-8 -*-
"""
Python 3.12 — Gera 4 tabelas a partir do CSV:
- dados_utilizados
- tecnicas_empregadas
- contribuica_trabalho
- dados_utilizados_combinados  <-- NOVA (combinações por artigo)

Todas com colunas: ["Classificação", "Número de Artigos"]

Regras:
- Divide itens por vírgula, ponto-e-vírgula, barra, pipe e quebras de linha.
- Remove acentos e caracteres especiais dos valores (mantém apenas letras, números e espaços).
- Dedup por artigo:
  * nas tabelas simples, cada classe conta no máximo 1 vez por artigo
  * na tabela de combinações, cada artigo conta exatamente 1 combinação (conjunto) dos seus itens
- Exporta CSVs para ./out se EXPORT_CSV=True
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


# ========== CONFIGURAÇÃO ==========
INPUT_CSV = r"./artigos_selecionados/artigos_selecionados_IEEE.csv"
EXPORT_CSV = True
OUTPUT_DIR = Path("./out")
# ==================================


def read_csv_safely(csv_path: str | Path) -> pd.DataFrame:
    """Lê CSV tentando com utf-8 e caindo para latin-1 se necessário."""
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


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


def split_items(cell: str) -> list[str]:
    """
    Divide uma célula em itens usando separadores comuns.
    NÃO divide por ' e ' para não quebrar nomes compostos.
    """
    if pd.isna(cell):
        return []
    parts = re.split(r"[,\n;/|]+", str(cell))
    return [p.strip() for p in parts if p and p.strip()]


def count_by_classification(series: pd.Series) -> pd.DataFrame:
    """
    Conta quantos artigos possuem cada classificação (1 por artigo por classe).
    - Faz split por célula (itens múltiplos contam para classes distintas)
    - Normaliza cada item removendo acento e caracteres especiais
    - Dedup por linha para evitar contar duas vezes a mesma classe no mesmo artigo
    """
    counter = Counter()
    for cell in series.astype(str).tolist():
        items = split_items(cell)
        norm_items = {normalize_strict_value(x) for x in items}  # set -> dedup por artigo
        norm_items = {x for x in norm_items if x}  # remove vazios
        counter.update(norm_items)

    df_out = (
        pd.DataFrame(
            {"Classificação": list(counter.keys()), "Número de Artigos": list(counter.values())}
        )
        .sort_values(["Número de Artigos", "Classificação"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df_out


def build_combination_label(items: set[str]) -> str:
    """
    Constrói um rótulo determinístico para a COMBINAÇÃO de itens:
    - Ordena alfabeticamente sem considerar caixa
    - Junta com ' | ' e aplica normalização estrita (removendo caracteres especiais)
      → o rótulo final contém somente letras, números e espaços.
    """
    if not items:
        return ""
    ordered = sorted(items, key=lambda s: s.casefold())
    raw = " | ".join(ordered)
    return normalize_strict_value(raw)


def count_by_combination(series: pd.Series) -> pd.DataFrame:
    """
    Conta quantos artigos pertencem a cada COMBINAÇÃO de 'Dados Utilizados'.
    - Para cada artigo:
        * separa itens
        * normaliza e deduplica
        * constrói um rótulo único da combinação (conjunto)
      Cada artigo contribui com exatamente 1 unidade para sua combinação.
    """
    counter = Counter()
    for cell in series.astype(str).tolist():
        items = split_items(cell)
        norm_items = {normalize_strict_value(x) for x in items if normalize_strict_value(x)}
        combo_label = build_combination_label(norm_items)
        if combo_label:
            counter.update([combo_label])

    df_out = (
        pd.DataFrame(
            {"Classificação": list(counter.keys()), "Número de Artigos": list(counter.values())}
        )
        .sort_values(["Número de Artigos", "Classificação"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df_out


def main():
    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = read_csv_safely(csv_path)

    # Localiza colunas alvo com tolerância a variações de nome
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

    # 3 tabelas originais
    dados_utilizados = count_by_classification(df[col_dados])
    tecnicas_empregadas = count_by_classification(df[col_tec])
    contribuica_trabalho = count_by_classification(df[col_contrib])

    # NOVA tabela de combinações em "Dados Utilizados"
    dados_utilizados_combinados = count_by_combination(df[col_dados])

    # Exibe no console
    print("\n=== dados_utilizados ===")
    print(dados_utilizados.to_string(index=False))

    print("\n=== tecnicas_empregadas ===")
    print(tecnicas_empregadas.to_string(index=False))

    print("\n=== contribuica_trabalho ===")
    print(contribuica_trabalho.to_string(index=False))

    print("\n=== dados_utilizados_combinados ===")
    print(dados_utilizados_combinados.to_string(index=False))

    # Exporta CSVs (opcional)
    if EXPORT_CSV:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dados_utilizados.to_csv(OUTPUT_DIR / "dados_utilizados.csv", index=False, encoding="utf-8")
        tecnicas_empregadas.to_csv(OUTPUT_DIR / "tecnicas_empregadas.csv", index=False, encoding="utf-8")
        contribuica_trabalho.to_csv(OUTPUT_DIR / "contribuica_trabalho.csv", index=False, encoding="utf-8")
        dados_utilizados_combinados.to_csv(
            OUTPUT_DIR / "dados_utilizados_combinados.csv", index=False, encoding="utf-8"
        )


if __name__ == "__main__":
    main()
