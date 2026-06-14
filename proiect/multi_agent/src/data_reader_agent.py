"""
Data Reader Agent — un singur graf LangGraph cu state tipat, noduri,
conditional edges și retry/fallback peste 3 surse: rag | csv | sql.

"""
import logging
import re
from operator import add
from pathlib import Path
from typing import Annotated, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "clienti.csv"


class DataReaderState(TypedDict, total=False):
    # input
    query: str
    max_retries: int
    # decizie LLM
    selected_source: str          # "rag" | "csv" | "sql"
    generated_query: str
    # tracking
    retry_count: int
    sources_tried: Annotated[list[str], add]
    # output
    result: Optional[dict]
    error: Optional[str]


class DataReaderAgent:
    """Agent cu o singură buclă de graf, peste rag/csv/sql, cu retry + fallback."""

    AVAILABLE_SOURCES = ("rag", "csv", "sql")

    def __init__(
        self,
        llm: LLMProvider | None = None,
        max_retries: int = 2,
        sql_table: str = "achizitii_directe",
    ):
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.max_retries = max_retries
        self.sql_table = sql_table
        self._sql_agent = None
        self._csv_df: pd.DataFrame | None = None
        self.graph = self.build_graph()


    def _exec_rag(self, query: str) -> tuple[Optional[dict], Optional[str]]:
        from database import transaction
        from rag_service import RAGService

        with transaction() as db:
            hits = RAGService(db).search(query, top_k=3, threshold=0.25)
        if not hits:
            return None, "RAG: niciun chunk relevant peste threshold."
        docs = [
            {"file_name": c.file_name, "content": c.content[:300], "score": round(s, 3)}
            for c, s in hits
        ]
        return {"source": "rag", "documents": docs}, None

    def _exec_sql(self, query: str) -> tuple[Optional[dict], Optional[str]]:
        if self._sql_agent is None:
            from nl2sql_agent import NL2SQLAgent
            d = DATA_DIR / "nl2sql_agent"
            self._sql_agent = NL2SQLAgent(
                table_name=self.sql_table,
                schema_path=str(d / f"schema_{self.sql_table}.json"),
                llm=self.llm,
                business_path=str(d / f"business_{self.sql_table}.json"),
            )
        res = self._sql_agent.run(query)
        if res.status != "success" or res.result is None or len(res.result) == 0:
            return None, f"SQL: {res.execution_error or res.validation_error or 'fără rezultate'}"
        return {
            "source": "sql",
            "sql": res.sql_query,
            "rows": res.result.head(10).to_dict(orient="records"),
        }, None

    # cuvinte prea generice ca să fie utile la căutare în CSV
    _STOPWORDS = {
        "care", "este", "are", "ce", "cum", "unde", "cine", "pentru", "despre",
        "are", "avem", "din", "cu", "pe", "la", "si", "și", "al", "ale", "un",
        "una", "numarul", "numărul", "datele", "informatii", "informații",
    }

    def _exec_csv(self, query: str) -> tuple[Optional[dict], Optional[str]]:
        if self._csv_df is None:
            if not CSV_PATH.exists():
                return None, "CSV: data/clienti.csv lipsește."
            self._csv_df = pd.read_csv(CSV_PATH)
        tokens = [
            t for t in re.findall(r"\w+", query.lower())
            if len(t) > 2 and t not in self._STOPWORDS
        ]
        if not tokens:
            return None, "CSV: query prea scurt pentru căutare."

        patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in tokens]
        haystack = self._csv_df.astype(str).apply(lambda r: " ".join(r).lower(), axis=1)
        mask = haystack.apply(lambda s: any(p.search(s) for p in patterns))
        hits = self._csv_df[mask]
        if hits.empty:
            return None, "CSV: nicio potrivire pentru termenii întrebării."
        return {"source": "csv", "rows": hits.to_dict(orient="records")}, None

    # ---------- noduri ----------

    def parse_query(self, state: DataReaderState) -> dict:
        return {"query": (state["query"] or "").strip()}

    def decide_source(self, state: DataReaderState) -> dict:
        """LLM alege sursa dintre cele neîncercate încă (reset retry pe sursa nouă)."""
        tried = set(state.get("sources_tried", []))
        candidates = [s for s in self.AVAILABLE_SOURCES if s not in tried] or list(self.AVAILABLE_SOURCES)
        prompt = self.prompts.render("datareader_decide", query=state["query"], sources=candidates)
        resp = self.llm.generate_sync([{"role": "user", "content": prompt}]).strip().lower()
        chosen = next((s for s in candidates if s in resp), candidates[0])
        logger.info(f"[DECIDE] candidați={candidates} → '{chosen}'")
        return {"selected_source": chosen, "retry_count": 0}

    def generate_query(self, state: DataReaderState) -> dict:
        return {"generated_query": state["query"]}

    def execute_query(self, state: DataReaderState) -> dict:
        src = state["selected_source"]
        q = state.get("generated_query") or state["query"]
        logger.info(f"[EXECUTE] sursa='{src}'")
        try:
            if src == "rag":
                result, error = self._exec_rag(q)
            elif src == "sql":
                result, error = self._exec_sql(q)
            elif src == "csv":
                result, error = self._exec_csv(q)
            else:
                result, error = None, f"Sursă necunoscută: {src}"
        except Exception as e:  
            result, error = None, f"{src}: {e}"

        update: dict = {"sources_tried": [src]}
        if error:
            update.update(
                result=None,
                error=error,
                retry_count=state.get("retry_count", 0) + 1,
            )
            logger.info(f"[EXECUTE] eșec: {error}")
        else:
            update.update(result=result, error=None)
            logger.info(f"[EXECUTE] succes pe '{src}'")
        return update

    def check_result(self, state: DataReaderState) -> dict:
        return {}

    # ---------- routere ----------

    def route_to_source(self, state: DataReaderState) -> str:
        return state["selected_source"] 

    def route_after_check(self, state: DataReaderState) -> str:
        if state.get("result") and not state.get("error"):
            return "success"
        if state.get("retry_count", 0) < state.get("max_retries", self.max_retries):
            return "retry"
        if set(state.get("sources_tried", [])) < set(self.AVAILABLE_SOURCES):
            return "fallback"
        return "fail"

    # ---------- graf ----------

    def build_graph(self):
        g = StateGraph(DataReaderState)
        g.add_node("parse", self.parse_query)
        g.add_node("decide", self.decide_source)
        g.add_node("generate", self.generate_query)
        g.add_node("execute", self.execute_query)
        g.add_node("check", self.check_result)

        g.add_edge(START, "parse")
        g.add_edge("parse", "decide")
        g.add_conditional_edges(
            "decide", self.route_to_source,
            {"rag": "generate", "csv": "generate", "sql": "generate"},
        )
        g.add_edge("generate", "execute")
        g.add_edge("execute", "check")
        g.add_conditional_edges(
            "check", self.route_after_check,
            {"success": END, "retry": "generate", "fallback": "decide", "fail": END},
        )
        return g.compile()

    def run(self, query: str, max_retries: int | None = None) -> dict:
        initial: DataReaderState = {
            "query": query,
            "max_retries": max_retries if max_retries is not None else self.max_retries,
            "retry_count": 0,
            "sources_tried": [],
        }
        return self.graph.invoke(initial)
