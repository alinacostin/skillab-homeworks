"""
Pydantic models pentru parametrii tool-urilor.

Convenție: toate tools primesc `input_dfs` (lista de DataFrames) + parametri specifici.
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class JoinDataParams(BaseModel):
    """Parametri pentru tool-ul join_data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dfs: list[pd.DataFrame] = Field(
        description="Lista de DataFrames: [left_df, right_df]"
    )
    left_key: str = Field(
        description="Coloana cheie din primul DataFrame"
    )
    right_key: str = Field(
        description="Coloana cheie din al doilea DataFrame"
    )
    how: Literal["inner", "left", "right", "outer"] = Field(
        default="inner",
        description="Tipul de join: 'inner', 'left', 'right', 'outer'"
    )


class FilterDataParams(BaseModel):
    """Parametri pentru tool-ul filter_data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dfs: list[pd.DataFrame] = Field(
        description="Lista de DataFrames: [df]"
    )
    column: str = Field(
        description="Coloana pe care se aplică filtrul"
    )
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains"] = Field(
        description="Operatorul de comparație"
    )
    value: str = Field(
        description="Valoarea pentru comparație"
    )

class DetectIntentParams(BaseModel):
    """Parametri pentru detect_intent."""
    query: str = Field(description="Întrebarea utilizatorului")


class ExtractFiltersParams(BaseModel):
    """Parametri pentru extract_filters."""
    query: str = Field(description="Întrebarea utilizatorului")


class AggregateResultsParams(BaseModel):
    """Parametri pentru aggregate_results."""
    data: str = Field(description="Înregistrările de agregat, ca array JSON (list[dict])")
    operations: list[str] = Field(
        default_factory=lambda: ["count"],
        description="Operațiile: count, sum, avg, min, max",
    )
    field: str = Field(default="", description="Coloana numerică pentru sum/avg/min/max")
    group_by: str | None = Field(default=None, description="Coloană opțională de grupare")


class FormatResponseParams(BaseModel):
    """Parametri pentru format_response."""
    data: str = Field(description="Conținutul de formatat")
    format_type: Literal["text", "table", "summary", "list"] = Field(
        default="summary", description="Tipul de formatare"
    )
    query: str = Field(default="", description="Întrebarea originală (context)")
