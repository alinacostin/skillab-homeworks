"""
Scheme Pydantic pentru tool-urile MCP — input/output validation (Tema 5).

Aici trăiește **input validation** cerută de temă: tip (`str`), dimensiune
(`max_length`) și câmpuri permise (`extra="forbid"` respinge orice cheie în plus).
Modelele de output proiectează state-urile agenților (care conțin DataFrame-uri și
obiecte Pydantic) într-o formă JSON-serializabilă pe care MCP o poate trimite.

Convenție: un `BaseModel` cu `Field(description=...)` per tool, din care FastMCP
derivează `inputSchema`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Plafon de dimensiune pe input (anti-abuz / anti-„bombă").
MAX_INPUT_CHARS = 4000


# ============================================================
# INPUT — validare tip / dimensiune / câmpuri permise
# ============================================================

class DataAnalystInput(BaseModel):
    """Input pentru tool-ul `data_analyst` — o întrebare analitică în limbaj natural."""
    model_config = ConfigDict(extra="forbid")  # câmpuri permise: doar `question`

    question: str = Field(
        description="Întrebarea analitică în limbaj natural (ro/en). Ex: 'Primii 10 "
                    "furnizori după valoarea totală a contractelor'.",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )


class OrchestratorInput(BaseModel):
    """Input pentru tool-ul `orchestrator` — o întrebare generală, rutată către workeri."""
    model_config = ConfigDict(extra="forbid")  # câmpuri permise: doar `query`

    query: str = Field(
        description="Întrebarea utilizatorului (ro/en). Supervizorul alege singur "
                    "workerii (rag/sql/csv) și agregă răspunsul.",
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )


# ============================================================
# OUTPUT — proiecție JSON-safe a state-urilor agenților
# ============================================================

class StepInfo(BaseModel):
    """Un pas din planul Analyst-ului (proiecție din `StepResult`)."""
    step_id: str
    action: str
    status: str
    row_count: int = 0
    error: str = ""


class DataAnalystOutput(BaseModel):
    """Rezultatul tool-ului `data_analyst`."""
    answer: str
    status: Literal["pending", "success", "failed", "no_plan", "error", "blocked"]
    plan: list[str] = Field(default_factory=list, description="ID-urile pașilor din plan")
    steps: list[StepInfo] = Field(default_factory=list)
    # populat doar pe căile de eroare/blocare
    error: str = ""
    reason: str = ""


class WorkerResult(BaseModel):
    """Rezultatul unui worker (rag/sql/csv) din Supervizor."""
    status: str
    answer: str = ""
    n_records: int = 0


class OrchestratorOutput(BaseModel):
    """Rezultatul tool-ului `orchestrator` (Supervizor)."""
    answer: str
    status: Literal["pending", "success", "partial", "failed", "error", "blocked"]
    intent: str = ""
    workers: list[str] = Field(default_factory=list)
    results: dict[str, WorkerResult] = Field(default_factory=dict)
    # populat doar pe căile de eroare/blocare
    error: str = ""
    reason: str = ""
