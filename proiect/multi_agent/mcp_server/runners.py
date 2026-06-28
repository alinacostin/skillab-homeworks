"""
Runners — instanțiere lazy a agenților + proiecție JSON-safe a state-urilor.

Aici e „handlerul care apelează agentul" cerut de temă (punctele 1 și 2). Agenții
sunt singletoni lazy: serverul pornește și listează tool-urile fără DB/LLM, iar
agentul se construiește abia la primul apel curat. State-urile lor conțin
DataFrame-uri și obiecte Pydantic → le proiectăm în dict-uri serializabile.

Rezolvarea provider/model din env oglindește `main.py` (alias gemini→google,
ollama→local; `*_MODEL` / `LLM_MODEL`).
"""
import _bootstrap  # noqa: F401  — pune `skillab-py/src` și `src/` pe sys.path

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MULTI_AGENT = Path(__file__).resolve().parent.parent
_NL2SQL_DIR = _MULTI_AGENT / "data" / "nl2sql_agent"

# ---- rezolvare provider/model (mirror main.py) ----
_PROVIDER_ALIASES = {"gemini": "google", "ollama": "local"}
_MODEL_ENV_VARS = {
    "google": "GOOGLE_MODEL", "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL", "local": "OLLAMA_MODEL",
}


def _resolve_provider(provider: str | None) -> str | None:
    return _PROVIDER_ALIASES.get(provider.lower(), provider.lower()) if provider else None


def _model_from_env(provider: str | None) -> str | None:
    resolved = _resolve_provider(provider)
    if not resolved:
        return None
    return os.getenv("LLM_MODEL") or os.getenv(_MODEL_ENV_VARS.get(resolved, f"{resolved.upper()}_MODEL"))


def _tables_config() -> dict:
    """Config NL2SQL pentru Analyst (mirror main.py:_tables_config)."""
    return {
        "achizitii_directe": {
            "schema_path": str(_NL2SQL_DIR / "schema_achizitii_directe.json"),
            "business_path": str(_NL2SQL_DIR / "business_achizitii_directe.json"),
        },
        "anunturi_initiere": {
            "schema_path": str(_NL2SQL_DIR / "schema_anunturi_initiere.json"),
            "business_path": str(_NL2SQL_DIR / "business_anunturi_initiere.json"),
        },
    }


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Citește un câmp fie din dict, fie din obiect Pydantic (LangGraph poate da oricare)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---- singletoni lazy ----
_llm = None
_analyst = None
_supervisor = None


def get_llm():
    global _llm
    if _llm is None:
        from skillab import get_llm as _factory
        provider = _resolve_provider(os.getenv("LLM_PROVIDER"))
        model = _model_from_env(os.getenv("LLM_PROVIDER"))
        _llm = _factory(provider=provider, model=model)
        logger.info("LLM: %s / %s", _llm.name, _llm.model)
    return _llm


def get_analyst():
    global _analyst
    if _analyst is None:
        from analyst_agent import AnalystAgent
        _analyst = AnalystAgent(tables_config=_tables_config(), llm=get_llm())
    return _analyst


def get_supervisor():
    global _supervisor
    if _supervisor is None:
        from supervisor import Supervisor
        _supervisor = Supervisor(llm=get_llm())
    return _supervisor


# ---- handlere: apel agent + proiecție JSON-safe ----

def run_data_analyst(question: str) -> dict:
    """Apelează `AnalystAgent.chat` și proiectează `AnalystState` în dict serializabil."""
    res = get_analyst().chat(question)
    plan = _attr(res, "plan", []) or []
    step_results = _attr(res, "step_results", []) or []
    return {
        "answer": _attr(res, "answer", ""),
        "status": _attr(res, "status", "failed"),
        "plan": [_attr(s, "id", "") for s in plan],
        "steps": [
            {
                "step_id": _attr(sr, "step_id", ""),
                "action": _attr(sr, "action", ""),
                "status": _attr(sr, "status", ""),
                "row_count": _attr(sr, "row_count", 0),
                "error": _attr(sr, "error", ""),
            }
            for sr in step_results
        ],
    }


def run_orchestrator(query: str) -> dict:
    """Apelează `Supervisor.run` și proiectează state-ul (agent_results) în dict serializabil."""
    res = get_supervisor().run(query)
    agent_results = _attr(res, "agent_results", {}) or {}
    results = {
        name: {
            "status": _attr(r, "status", ""),
            "answer": _attr(r, "answer", ""),
            "n_records": len(_attr(r, "records", []) or []),
        }
        for name, r in agent_results.items()
    }
    return {
        "answer": _attr(res, "answer", ""),
        "status": _attr(res, "status", "failed"),
        "intent": _attr(res, "intent", ""),
        "workers": _attr(res, "workers", []) or [],
        "results": results,
    }
