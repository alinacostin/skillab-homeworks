"""
Server MCP — expune agenții ca tool-uri + guardrails (Tema 5).

Două tool-uri în același server (FastMCP, transport stdio):
  • `data_analyst`  → AnalystAgent.chat   (plan → query → tool → synthesize)
  • `orchestrator`  → Supervisor.run      (detect_intent → workeri → aggregate)

Fluxul fiecărui tool (ordinea contează):
  1. INPUT VALIDATION — modelul Pydantic verifică tip / dimensiune / câmpuri permise.
     Eșec → McpError -32602 (invalid params).
  2. GUARDRAIL prompt-injection — InputValidator (regex + opțional embedding / LLM-judge).
     Input periculos → refuz STRUCTURAT (status="blocked"), agentul NU e apelat.
  3. APEL AGENT — eșec de infrastructură → McpError -32000 (server error).

Rulare:
  python server.py            # stdio (default) — pentru Claude Code / Claude Desktop
  python server.py http       # http://127.0.0.1:8000  (debug din browser/curl)

NB: transportul stdio folosește stdout pentru protocol → logăm DOAR pe stderr
(logging.basicConfig scrie implicit pe stderr) și nu folosim `print`.
"""
import _bootstrap  # noqa: F401  — pune `skillab-py/src` și `src/` pe sys.path

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from pydantic import ValidationError

import runners
from guardrails import InputValidator
from schemas import DataAnalystInput, DataAnalystOutput, OrchestratorInput, OrchestratorOutput

# Log pe stderr; reducem zgomotul (ca în main.py) — stdout e rezervat protocolului MCP.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S", stream=sys.stderr,
)
for _n in ("httpx", "sentence_transformers", "transformers", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logger = logging.getLogger("mcp_server")

# Coduri JSON-RPC.
INVALID_PARAMS = -32602
SERVER_ERROR = -32000

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

mcp = FastMCP("document-agents")


# ---------- guardrail (lazy: straturile scumpe doar dacă-s cerute) ----------

_validator: InputValidator | None = None


def _get_validator() -> InputValidator:
    global _validator
    if _validator is None:
        use_emb = os.getenv("GUARDRAIL_EMBEDDING", "0") == "1"
        use_llm = os.getenv("GUARDRAIL_LLM_JUDGE", "0") == "1"
        llm = runners.get_llm() if use_llm else None
        prompts = None
        if use_llm:
            from skillab.prompts import PromptRegistry
            prompts = PromptRegistry(str(PROMPTS_DIR))
        _validator = InputValidator(use_embedding=use_emb, use_llm=use_llm, llm=llm, prompts=prompts)
        logger.info("Guardrails: regex=on embedding=%s llm_judge=%s", use_emb, use_llm)
    return _validator


class _Blocked(Exception):
    """Marker intern: input respins de guardrail (refuz structurat, nu eroare)."""
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _validate_and_guard(input_model, field: str, value: str) -> str:
    """Pasul 1 (validare) + pasul 2 (guardrail). Întoarce textul curat sau ridică."""
    # 1. input validation — tip / dimensiune / câmpuri permise
    try:
        validated = input_model(**{field: value})
    except ValidationError as e:
        msg = e.errors()[0].get("msg", "input invalid") if e.errors() else "input invalid"
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Input invalid: {msg}"))
    text = getattr(validated, field)
    # 2. guardrail prompt-injection
    verdict = _get_validator().validate(text)
    if not verdict.passed:
        raise _Blocked(verdict.reason)
    return text


# ---------- TOOL 1: Data Analyst ----------

@mcp.tool()
def data_analyst(question: str) -> DataAnalystOutput:
    """Agent Data Analyst: plan multi-pas peste tabelele de achiziții (NL2SQL +
    join/filter/sort) și sinteză a răspunsului. Folosește pentru întrebări analitice
    pe date structurate (top furnizori, agregări, filtrări)."""
    try:
        text = _validate_and_guard(DataAnalystInput, "question", question)
    except _Blocked as b:
        logger.info("[data_analyst] BLOCKED: %s", b.reason)
        return DataAnalystOutput(answer="", status="blocked", reason=b.reason)
    try:
        result = runners.run_data_analyst(text)
    except Exception as e:  # noqa: BLE001 — eșec infra → McpError -32000
        logger.exception("[data_analyst] eșec agent")
        raise McpError(ErrorData(code=SERVER_ERROR, message=f"{type(e).__name__}: {str(e)[:200]}"))
    return DataAnalystOutput(**result)


# ---------- TOOL 2: Orchestrator / Supervisor ----------

@mcp.tool()
def orchestrator(query: str) -> OrchestratorOutput:
    """Agent Orchestrator/Supervizor: detectează intentul, alege singur workerii
    (rag pe documente / sql pe achiziții / csv pe furnizori) și agregă răspunsul.
    Folosește pentru întrebări generale sau care combină mai multe surse."""
    try:
        text = _validate_and_guard(OrchestratorInput, "query", query)
    except _Blocked as b:
        logger.info("[orchestrator] BLOCKED: %s", b.reason)
        return OrchestratorOutput(answer="", status="blocked", reason=b.reason)
    try:
        result = runners.run_orchestrator(text)
    except Exception as e:  # noqa: BLE001 — eșec infra → McpError -32000
        logger.exception("[orchestrator] eșec agent")
        raise McpError(ErrorData(code=SERVER_ERROR, message=f"{type(e).__name__}: {str(e)[:200]}"))
    return OrchestratorOutput(**result)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run(transport="http", host="127.0.0.1", port=8000)
    else:
        mcp.run()  # stdio (default)
