"""
NL2SQL Agent - TODO: Implementează nodurile
"""
import json
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlparse
from langgraph.graph import StateGraph, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from state import NL2SQLState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class NL2SQLAgent:
    """
    Flow:
        get_context → generate_sql → validate → execute
                                        ↓
                                    handle_error
    """

    def __init__(
        self,
        table_name: str,
        schema_path: str,
        max_retries: int = 2,
        llm: LLMProvider | None = None,
        business_path: str | None = None,
    ):
        self.table_name = table_name
        self.max_retries = max_retries
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load schema + (opțional) regulile business care ajută generarea SQL
        self.schema = json.loads(Path(schema_path).read_text())
        self.business_rules = {}
        if business_path:
            self.business_rules = json.loads(Path(business_path).read_text()).get("rules", {})

        self.graph = self._build_graph()

    @staticmethod
    def _clean_sql(text: str) -> str:
        """Scoate fence-urile ```sql ... ``` și punctuația de final dintr-un răspuns LLM."""
        text = (text or "").strip()
        fenced = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        return text.rstrip(";").strip()

    # === NODES ===

    def node_get_context(self, state: NL2SQLState) -> dict:
        """COMPLET."""
        return {
            "schema_context": self.schema,
            "table_name": self.table_name,
        }

    def node_generate_sql(self, state: NL2SQLState) -> dict:
        """Generează SQL (SELECT) din întrebare folosind schema + regulile business."""
        logger.info(f"[GENERATE] {state.question}")

        prompt = self.prompts.render(
            "nl2sql_generate",
            table_name=self.table_name,
            table_description=self.schema.get("description", ""),
            columns=self.schema.get("columns", {}),
            business_rules=self.business_rules,
            question=state.question,
        )
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])
        sql = self._clean_sql(response)
        logger.info(f"[GENERATE] SQL: {sql[:120]}")
        return {"sql_query": sql}

    # cuvinte care nu au ce căuta într-un SELECT read-only
    _FORBIDDEN = (
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
        "CREATE", "GRANT", "REVOKE", "MERGE", "COPY", "REPLACE",
    )

    def node_validate_sql(self, state: NL2SQLState) -> dict:
        """Validează SQL: doar un singur SELECT, fără cuvinte periculoase/injection."""
        sql = (state.sql_query or "").strip()
        logger.info(f"[VALIDATE] {sql[:60]}...")

        if not sql:
            return {"is_valid": False, "validation_error": "SQL gol."}

        parsed = sqlparse.parse(sql)
        statements = [s for s in parsed if str(s).strip()]
        if not statements:
            return {"is_valid": False, "validation_error": "Nu s-a putut parsa SQL-ul."}
        if len(statements) > 1:
            return {"is_valid": False, "validation_error": "Sunt permise doar query-uri singulare (un singur SELECT)."}

        stmt_type = statements[0].get_type()
        is_select = stmt_type == "SELECT" or sql.upper().lstrip().startswith("WITH")
        if not is_select:
            return {"is_valid": False, "validation_error": f"Doar SELECT este permis (detectat: {stmt_type})."}

        upper = sql.upper()
        found = [kw for kw in self._FORBIDDEN if re.search(rf"\b{kw}\b", upper)]
        if found:
            return {"is_valid": False, "validation_error": f"Cuvinte interzise în query: {', '.join(found)}."}
        if "--" in sql or "/*" in sql:
            return {"is_valid": False, "validation_error": "Comentariile SQL nu sunt permise."}

        return {"is_valid": True, "validation_error": ""}

    def node_execute_sql(self, state: NL2SQLState) -> dict:
        """Execută SELECT-ul și întoarce un DataFrame. Pe eroare, setează execution_error."""
        from sqlalchemy import text
        from database import transaction

        sql = state.sql_query
        # plasă de siguranță: evită materializarea a sute de mii de rânduri
        if not re.search(r"\blimit\b", sql, re.IGNORECASE):
            sql = f"{sql} LIMIT 1000"

        logger.info("[EXECUTE]")
        try:
            with transaction() as session:
                result = session.execute(text(sql))
                df = pd.DataFrame(result.mappings().all())
            logger.info(f"[EXECUTE] OK — {len(df)} rânduri")
            return {"result": df, "execution_error": "", "status": "success"}
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            logger.warning(f"[EXECUTE] eroare: {msg}")
            return {"result": pd.DataFrame(), "execution_error": msg}

    def node_handle_error(self, state: NL2SQLState) -> dict:
        """Incrementează retry-ul; dacă mai avem încercări, cere LLM-ului SQL corectat."""
        new_retry = state.retry_count + 1
        error_msg = state.execution_error or state.validation_error or "(necunoscut)"
        logger.info(f"[ERROR] retry {new_retry}/{self.max_retries} — {error_msg[:80]}")

        if new_retry > self.max_retries:
            logger.info("[ERROR] max_retries atins → renunț")
            return {"retry_count": new_retry, "status": "failed"}

        prompt = self.prompts.render(
            "nl2sql_error",
            table_name=self.table_name,
            question=state.question,
            failed_sql=state.sql_query,
            error_message=error_msg,
            columns=self.schema.get("columns", {}),
        )
        corrected = self._clean_sql(self.llm.generate_sync([{"role": "user", "content": prompt}]))
        logger.info(f"[ERROR] SQL corectat: {corrected[:120]}")
        return {
            "sql_query": corrected,
            "retry_count": new_retry,
            "execution_error": "",
            "validation_error": "",
            "status": "pending",
        }

    # === ROUTING ===

    def _route_after_validate(self, state: NL2SQLState) -> str:
        return "execute_sql" if state.is_valid else "handle_error"

    def _route_after_execute(self, state: NL2SQLState) -> str:
        return END if not state.execution_error else "handle_error"

    def _route_after_error(self, state: NL2SQLState) -> str:
        # handle_error decide când renunță (status="failed"); altfel reîncercăm
        return END if state.status == "failed" else "generate_sql"

    # === GRAPH ===

    def _build_graph(self):
        graph = StateGraph(NL2SQLState)

        graph.add_node("get_context", self.node_get_context)
        graph.add_node("generate_sql", self.node_generate_sql)
        graph.add_node("validate_sql", self.node_validate_sql)
        graph.add_node("execute_sql", self.node_execute_sql)
        graph.add_node("handle_error", self.node_handle_error)

        graph.set_entry_point("get_context")
        graph.add_edge("get_context", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_conditional_edges("validate_sql", self._route_after_validate, ["execute_sql", "handle_error"])
        graph.add_conditional_edges("execute_sql", self._route_after_execute, [END, "handle_error"])
        graph.add_conditional_edges("handle_error", self._route_after_error, ["generate_sql", END])

        return graph.compile()

    def run(self, question: str) -> NL2SQLState:
        """Execută agentul.

        LangGraph întoarce state-ul ca dict; îl reconstruim tipat ca Analyst-ul
        (`_execute_query`) să poată accesa `.status`, `.result`, `.execution_error`.
        """
        initial = NL2SQLState(
            question=question,
            table_name=self.table_name,
            max_retries=self.max_retries,
        )
        final = self.graph.invoke(initial)
        return NL2SQLState(**final)
