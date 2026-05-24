"""Pydantic params models per tool.

Fiecare tool primește un singur BaseModel cu câmpuri validate.
Descrierile din Field(...) ajung la LLM ca documentație în catalog.
"""

from pydantic import BaseModel, Field


class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Expresia matematică de evaluat (ex: '2 + 3 * 4'). Suportă +, -, *, /, **, paranteze.",
        min_length=1,
    )


class GetDateTimeParams(BaseModel):
    timezone: str = Field(
        default="Europe/Bucharest",
        description="IANA timezone name (ex: 'Europe/Bucharest', 'UTC', 'America/New_York').",
        min_length=1,
    )


class WebSearchParams(BaseModel):
    query: str = Field(
        description="Termenii de căutat pe web.",
        min_length=2,
    )
    max_results: int = Field(
        default=5,
        description="Numărul maxim de rezultate returnate.",
        ge=1,
        le=10,
    )
