"""
Supervisor — agentul central care leagă tot.

Pattern Supervisor: un graf de nivel înalt care
  1. preprocess  — `detect_intent` + `extract_filters` (tool-uri @register_tool) →
                   alege worker(i) dintre rag/sql/csv
  2. rulează worker(i), shared state cu reducer pe `agent_results`:
       - rag → Orchestrator (RAG cu evaluate/refine)
       - sql → AnalystAgent (plan → NL2SQL → join/filter → synthesize)
       - csv → DataReaderAgent (Data Reader-ul wrap-uit ca un worker)
  3. postprocess — `aggregate_results` + `format_response` (tool-uri) → răspuns final

Flow (secvențial — generate_sync folosește un event loop partajat, deci workerii nu
pot rula concurent; reducer-ul pe agent_results e exercitat fiindcă fiecare worker
scrie în el):

    START → preprocess → rag_worker → sql_worker → csv_worker → aggregate → END
                          (fiecare worker e no-op dacă nu e selectat)
"""
import json
import logging
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.tools import ToolWrapper

from orchestrator import Orchestrator, OrchestratorConfig
from analyst_agent import AnalystAgent
from data_reader_agent import DataReaderAgent
from state import OrchestratorState

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def _merge_results(a: dict, b: dict) -> dict:
    """Reducer pentru agent_results: unește dict-urile scrise de workeri."""
    out = dict(a or {})
    out.update(b or {})
    return out


class SupervisorState(TypedDict, total=False):
    query: str
    intent: str
    filters: dict
    workers: list[str]
    agent_results: Annotated[dict, _merge_results]   # worker_name -> {answer, status, records, ...}
    answer: str
    status: str


class Supervisor:
    """Agent central care alege și coordonează workerii (rag / sql / csv)."""

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm()
        self._orch: Orchestrator | None = None
        self._analyst: AnalystAgent | None = None
        self._reader: DataReaderAgent | None = None
        self.graph = self.build_graph()

    # ---------- workeri (lazy) ----------

    def _orchestrator(self) -> Orchestrator:
        if self._orch is None:
            cfg = OrchestratorConfig()
            cfg.max_iterations = 1  # supervisor-ul ține bucla RAG scurtă
            self._orch = Orchestrator(config=cfg, llm=self.llm)
        return self._orch

    def _analyst_agent(self) -> AnalystAgent:
        if self._analyst is None:
            d = DATA_DIR / "nl2sql_agent"
            self._analyst = AnalystAgent(
                tables_config={
                    "achizitii_directe": {
                        "schema_path": str(d / "schema_achizitii_directe.json"),
                        "business_path": str(d / "business_achizitii_directe.json"),
                    },
                    "anunturi_initiere": {
                        "schema_path": str(d / "schema_anunturi_initiere.json"),
                        "business_path": str(d / "business_anunturi_initiere.json"),
                    },
                },
                llm=self.llm,
            )
        return self._analyst

    def _data_reader(self) -> DataReaderAgent:
        # DataReaderAgent wrap-uit ca worker (cerința „wrap agentul Data Reader ca un worker").
        if self._reader is None:
            self._reader = DataReaderAgent(llm=self.llm)
        return self._reader

    # ---------- noduri ----------

    def node_preprocess(self, state: SupervisorState) -> dict:
        """Preprocess: detect_intent + extract_filters (tool-uri) → alege workerii."""
        q = state["query"]
        intent = json.loads(ToolWrapper.call("detect_intent", {"query": q}))
        filters = json.loads(ToolWrapper.call("extract_filters", {"query": q}))
        workers = [s for s in intent.get("sources", []) if s in ("rag", "sql", "csv")] or ["rag"]
        logger.info(f"[PREPROCESS] intent={intent['intent']} workers={workers} filters={filters}")
        return {"intent": intent["intent"], "filters": filters, "workers": workers}

    def node_rag_worker(self, state: SupervisorState) -> dict:
        if "rag" not in state.get("workers", []):
            return {}
        res = self._orchestrator().build_graph().invoke(OrchestratorState(query=state["query"]))
        rr = res.get("rag_result")
        sources = sorted({r.file_name for r in rr.results}) if rr else []
        logger.info(f"[WORKER rag] status={res['status']} sources={sources}")
        return {"agent_results": {"rag": {
            "answer": res["answer"], "status": res["status"], "sources": sources, "records": [],
        }}}

    def node_sql_worker(self, state: SupervisorState) -> dict:
        if "sql" not in state.get("workers", []):
            return {}
        res = self._analyst_agent().chat(state["query"])
        # atașează rândurile finale ca records (pentru aggregate_results în postprocess)
        records = []
        plan = res.get("plan") or []
        slices = res.get("slices") or {}
        if plan:
            final = slices.get(plan[-1].id)
            if final is not None and len(final):
                records = final.head(10).to_dict("records")
        logger.info(f"[WORKER sql] status={res['status']} records={len(records)}")
        return {"agent_results": {"sql": {
            "answer": res["answer"], "status": res["status"], "records": records,
        }}}

    def node_csv_worker(self, state: SupervisorState) -> dict:
        # csv worker = DataReaderAgent, wrap-uit ca worker
        if "csv" not in state.get("workers", []):
            return {}
        res = self._data_reader().run(state["query"])
        result = res.get("result")
        if not result:
            logger.info(f"[WORKER csv] fără rezultat ({res.get('error')})")
            return {"agent_results": {"csv": {
                "answer": f"(fără rezultat: {res.get('error')})", "status": "failed", "records": [],
            }}}
        records = result.get("rows", []) or []
        logger.info(f"[WORKER csv] sursă={result.get('source')} records={len(records)}")
        return {"agent_results": {"csv": {
            "answer": self._format_reader_result(result), "status": "success",
            "records": records, "source": result.get("source"),
        }}}

    @staticmethod
    def _format_reader_result(result: dict) -> str:
        src = result.get("source")
        if result.get("rows"):
            lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in result["rows"][:10]]
            return f"(sursă {src})\n" + "\n".join(lines)
        if result.get("documents"):
            docs = result["documents"][:3]
            return f"(sursă {src}) " + "; ".join(f"{d['file_name']}: {d['content'][:80]}" for d in docs)
        return f"(sursă {src})"

    def node_aggregate(self, state: SupervisorState) -> dict:
        """Postprocess: aggregate_results (numără înregistrările) + format_response."""
        results = state.get("agent_results", {})
        if not results:
            return {"answer": "Niciun worker nu a produs un rezultat.", "status": "failed"}

        # postprocess tool 1 — aggregate_results peste înregistrările tabelare ale workerilor
        records = []
        for r in results.values():
            records += r.get("records", []) or []
        meta = ""
        if records:
            agg = json.loads(ToolWrapper.call("aggregate_results", {
                "data": json.dumps(records, default=str, ensure_ascii=False),
                "operations": ["count"], "field": "",
            }))
            meta = f"\n\n_({agg.get('count', 0)} înregistrări analizate)_"

        # conținut: un singur worker → text; mai mulți → secțiuni
        if len(results) == 1:
            content = next(iter(results.values()))["answer"] + meta
            fmt = "text"
        else:
            content = "\n\n".join(f"### {name}\n{r['answer']}" for name, r in results.items()) + meta
            fmt = "summary"

        # postprocess tool 2 — format_response
        answer = ToolWrapper.call("format_response", {
            "data": content, "format_type": fmt, "query": state["query"],
        })
        statuses = [r.get("status") for r in results.values()]
        if any(s == "success" for s in statuses):
            status = "success"
        elif any(s == "partial" for s in statuses):
            status = "partial"
        else:
            status = "failed"
        logger.info(f"[AGGREGATE] {list(results)} → status={status}")
        return {"answer": answer, "status": status}

    # ---------- graf ----------

    def build_graph(self):
        g = StateGraph(SupervisorState)
        g.add_node("preprocess", self.node_preprocess)
        g.add_node("rag_worker", self.node_rag_worker)
        g.add_node("sql_worker", self.node_sql_worker)
        g.add_node("csv_worker", self.node_csv_worker)
        g.add_node("aggregate", self.node_aggregate)

        g.add_edge(START, "preprocess")
        g.add_edge("preprocess", "rag_worker")
        g.add_edge("rag_worker", "sql_worker")
        g.add_edge("sql_worker", "csv_worker")
        g.add_edge("csv_worker", "aggregate")
        g.add_edge("aggregate", END)
        return g.compile()

    def run(self, query: str) -> dict:
        return self.graph.invoke({"query": query})
