"""
Tool-uri de pre/postprocess pentru Analyst / Supervisor:

  preprocess:  detect_intent, extract_filters
  postprocess: aggregate_results, format_response

"""
import json
import re

from .registry import register_tool
from .params import (
    DetectIntentParams,
    ExtractFiltersParams,
    AggregateResultsParams,
    FormatResponseParams,
)

# reguli keyword → sursă (varianta rule-based)
_SOURCE_KEYWORDS: dict[str, list[str]] = {
    "sql": [
        "achizit", "furnizor", "valoare", "valoarea", "total", "câte", "cati",
        "câți", "numar", "număr", "top", "primii", "primele", "suma", "medie",
        "autoritate", "câștigător", "castigator",
    ],
    "csv": ["email", "telefon", "contact", "domeniu", "oraș", "oras", "adresa", "adresă"],
    "rag": [
        "contract", "contractul", "document", "documentul", "clauz", "conține",
        "contine", "spune", "raport", "rezili", "prevede", "factura", "factură",
    ],
}
_INTENT_BY_SOURCE = {"sql": "aggregate", "rag": "search", "csv": "lookup"}

# companii cunoscute din corpus (pentru extract_filters)
_KNOWN_COMPANIES = ["TechSoft", "DataPro", "CloudNet", "SecureIT", "WebDev"]


@register_tool
def detect_intent(params: DetectIntentParams) -> str:
    """Detectează intentul întrebării și sursele potrivite (rag/sql/csv), rule-based.
    Întoarce JSON: {intent, confidence, suggested_source, sources}."""
    q = params.query.lower()
    matched = [src for src, kws in _SOURCE_KEYWORDS.items() if any(k in q for k in kws)]
    if not matched:
        result = {"intent": "search", "confidence": 0.3, "suggested_source": "rag", "sources": ["rag"]}
    else:
        sources = [s for s in ("sql", "rag", "csv") if s in matched]
        primary = sources[0]
        result = {
            "intent": _INTENT_BY_SOURCE[primary],
            "confidence": 0.9,
            "suggested_source": primary,
            "sources": sources,
        }
    return json.dumps(result, ensure_ascii=False)


@register_tool
def extract_filters(params: ExtractFiltersParams) -> str:
    """Extrage filtre din întrebare (companie, prag valoric), rule-based. Întoarce JSON."""
    q = params.query
    filters: dict = {}

    for comp in _KNOWN_COMPANIES:
        if comp.lower() in q.lower():
            filters["company"] = comp
            break

    m_min = re.search(r"(?:peste|mai mare de|mai mult de|>)\s*([\d.]+)", q, re.IGNORECASE)
    if m_min:
        filters["amount_min"] = float(m_min.group(1).replace(".", ""))
    m_max = re.search(r"(?:sub|mai mic de|mai puțin de|<)\s*([\d.]+)", q, re.IGNORECASE)
    if m_max:
        filters["amount_max"] = float(m_max.group(1).replace(".", ""))

    return json.dumps(filters, ensure_ascii=False)


@register_tool
def aggregate_results(params: AggregateResultsParams) -> str:
    """Agregă o listă JSON de înregistrări: count / sum / avg / min / max, opțional group_by.
    Întoarce JSON cu rezultatele agregării."""
    try:
        data = json.loads(params.data)
    except (json.JSONDecodeError, TypeError):
        data = []
    if not isinstance(data, list):
        data = []

    def _agg(rows: list[dict]) -> dict:
        out: dict = {}
        if "count" in params.operations:
            out["count"] = len(rows)
        if params.field:
            vals = []
            for row in rows:
                if isinstance(row, dict):
                    try:
                        vals.append(float(row.get(params.field)))
                    except (TypeError, ValueError):
                        pass
            if vals:
                if "sum" in params.operations:
                    out["sum"] = sum(vals)
                if "avg" in params.operations:
                    out["avg"] = sum(vals) / len(vals)
                if "min" in params.operations:
                    out["min"] = min(vals)
                if "max" in params.operations:
                    out["max"] = max(vals)
        return out

    if params.group_by:
        groups: dict[str, list[dict]] = {}
        for row in data:
            key = str(row.get(params.group_by, "—")) if isinstance(row, dict) else "—"
            groups.setdefault(key, []).append(row)
        result = {k: _agg(v) for k, v in groups.items()}
    else:
        result = _agg(data)

    return json.dumps(result, ensure_ascii=False)


@register_tool
def format_response(params: FormatResponseParams) -> str:
    """Formatează un răspuns pentru utilizator (text | table | summary | list).
    Întoarce string-ul formatat (nu JSON)."""
    data = params.data
    ft = params.format_type
    if ft == "summary":
        return f"## Rezumat\n\n{data}"
    if ft == "list":
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        return "\n".join(f"- {ln}" for ln in lines)
    if ft == "table":
        return f"```\n{data}\n```"
    return data  # text
