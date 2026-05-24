"""Implementări concrete pentru tool-uri: calculator, get_datetime, web_search.

Fiecare funcție:
  - primește un BaseModel din params_models
  - are docstring >= 15 caractere (devine description pentru LLM)
  - returnează un string (consumat ca ToolMessage.content)
"""

import ast
import operator as op
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .params_models import CalculatorParams, GetDateTimeParams, WebSearchParams
from .registry import register_tool

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Expresie nepermisă: {ast.dump(node)}")


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluează o expresie matematică cu operatori +, -, *, /, **, paranteze.
    Nu execută cod arbitrar — doar aritmetică pe numere."""
    tree = ast.parse(params.expression, mode="eval")
    rezultat = _safe_eval(tree)
    if isinstance(rezultat, float) and rezultat.is_integer():
        rezultat = int(rezultat)
    return str(rezultat)


@register_tool
def get_datetime(params: GetDateTimeParams) -> str:
    """Returnează data și ora curentă în formatul ISO 8601, pentru un timezone IANA dat."""
    try:
        tz = ZoneInfo(params.timezone)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Timezone necunoscut: {params.timezone}") from e
    return datetime.now(tz).isoformat(timespec="seconds")


@register_tool
def web_search(params: WebSearchParams) -> str:
    """Caută pe web prin DuckDuckGo și returnează primele rezultate ca linii 'titlu — snippet — url'."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise RuntimeError(
            "Pachetul 'ddgs' nu e instalat. Rulează: pip install ddgs"
        ) from e

    with DDGS() as ddgs:
        hits = list(ddgs.text(params.query, max_results=params.max_results))

    if not hits:
        return "Niciun rezultat."

    linii = []
    for h in hits:
        titlu = h.get("title", "").strip()
        snippet = h.get("body", "").strip().replace("\n", " ")
        url = h.get("href", "").strip()
        linii.append(f"- {titlu} — {snippet} — {url}")
    return "\n".join(linii)
