"""
Tool implementations — funcțiile efective ale tool-urilor.

Convenție: toate tools primesc params cu `input_dfs` (lista de DataFrames) + parametri specifici.
Tool-urile pot ridica excepții — Analyst-ul (`_execute_tool`) le prinde și le transformă
în `StepResult(status="failed")`, deci nu omoară graful.
"""
import pandas as pd

from .registry import register_tool
from .params import JoinDataParams, FilterDataParams


@register_tool
def join_data(params: JoinDataParams) -> pd.DataFrame:
    """
    Combină două DataFrames pe baza unei chei comune (join).
    Suportă inner, left, right, outer join.

    Args:
        params.input_dfs: [left_df, right_df]
        params.left_key: coloana cheie din primul DataFrame
        params.right_key: coloana cheie din al doilea DataFrame
        params.how: tipul de join (inner | left | right | outer)

    Returns:
        DataFrame rezultat după join
    """
    if len(params.input_dfs) < 2:
        raise ValueError(
            f"join_data necesită 2 DataFrames în input_dfs, am primit {len(params.input_dfs)}"
        )

    left_df, right_df = params.input_dfs[0], params.input_dfs[1]

    if params.left_key not in left_df.columns:
        raise KeyError(
            f"Coloana left_key '{params.left_key}' nu există în primul DataFrame. "
            f"Disponibile: {list(left_df.columns)}"
        )
    if params.right_key not in right_df.columns:
        raise KeyError(
            f"Coloana right_key '{params.right_key}' nu există în al doilea DataFrame. "
            f"Disponibile: {list(right_df.columns)}"
        )

    return pd.merge(
        left_df,
        right_df,
        left_on=params.left_key,
        right_on=params.right_key,
        how=params.how,
        suffixes=("_left", "_right"),
    )


@register_tool
def filter_data(params: FilterDataParams) -> pd.DataFrame:
    """
    Filtrează un DataFrame pe baza unei condiții.
    Suportă operatori: ==, !=, >, <, >=, <=, contains.

    Valoarea vine ca string; pentru operatorii de ordine (>, <, >=, <=) și pentru
    ==/!= se încearcă întâi comparația numerică, apoi cea pe text.

    Args:
        params.input_dfs: [df]
        params.column: coloana pe care se aplică filtrul
        params.operator: operatorul de comparație
        params.value: valoarea pentru comparație (string)

    Returns:
        DataFrame filtrat
    """
    if not params.input_dfs:
        raise ValueError("filter_data necesită un DataFrame în input_dfs")

    df = params.input_dfs[0]
    if params.column not in df.columns:
        raise KeyError(
            f"Coloana '{params.column}' nu există. Disponibile: {list(df.columns)}"
        )

    col = df[params.column]
    op = params.operator
    raw = params.value

    # contains → mereu pe text, case-insensitive
    if op == "contains":
        mask = col.astype(str).str.contains(str(raw), case=False, na=False)
        return df[mask]

    # încearcă comparație numerică dacă valoarea e numerică
    try:
        num_val = float(raw)
    except (TypeError, ValueError):
        num_val = None

    if num_val is not None:
        numeric_col = pd.to_numeric(col, errors="coerce")
        cmp = {
            "==": numeric_col == num_val,
            "!=": numeric_col != num_val,
            ">": numeric_col > num_val,
            "<": numeric_col < num_val,
            ">=": numeric_col >= num_val,
            "<=": numeric_col <= num_val,
        }[op]
        return df[cmp.fillna(False)]

    # fallback pe text (doar egalitate are sens pentru valori ne-numerice)
    if op == "==":
        return df[col.astype(str) == str(raw)]
    if op == "!=":
        return df[col.astype(str) != str(raw)]

    raise ValueError(
        f"Operatorul '{op}' necesită o valoare numerică, dar '{raw}' nu poate fi convertit."
    )
