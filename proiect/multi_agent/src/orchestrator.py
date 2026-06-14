"""
Orchestrator - Supervizor care coordonează RAG Agent

Flow:
    call_rag → evaluate ──┬──→ answer → END
         ↑                │
         │   can_answer   │
         │   = false      │
         └────────────────┘

TODO pentru studenți: node_evaluate, node_answer
"""
import logging
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from json_utils import extract_json
from state import OrchestratorState, OrchestratorFeedback
from rag_agent import RAGAgent, RAGAgentConfig

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OrchestratorConfig:
    max_iterations: int = 3
    min_score: float = 0.25


class Orchestrator:
    """
    Supervizor care coordonează RAG Agent.

    Flow:
        1. Apelează RAG Agent pentru căutare
        2. Evaluează: pot răspunde cu aceste chunks?
        3. Dacă DA → generează răspuns
        4. Dacă NU → trimite feedback la RAG Agent, repeat
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # RAG Agent - graf separat
        rag_config = RAGAgentConfig()
        rag_config.default_threshold = self.config.min_score
        self.rag = RAGAgent(self.llm, rag_config)

    # === NODES ===

    def node_call_rag(self, state: OrchestratorState) -> dict:
        """Apelează RAG Agent pentru căutare. COMPLET - nu modifica."""
        logger.info(f"[CALL_RAG] iter {state.iteration + 1}")

        # Apelează RAG Agent cu query și feedback (dacă există)
        rag_result = self.rag.run(
            query=state.query,
            feedback=state.feedback,  # None prima dată
        )

        return {
            "rag_result": rag_result.result,
            "iteration": state.iteration + 1,
        }

    @staticmethod
    def _build_context(state: OrchestratorState) -> str:
        """Concatenează chunk-urile găsite într-un context citabil (cu sursa)."""
        if not state.rag_result or not state.rag_result.results:
            return ""
        return "\n\n".join(
            f"[{r.file_name}]\n{r.content}" for r in state.rag_result.results
        )

    def node_evaluate(self, state: OrchestratorState) -> dict:
        """
        Evaluează dacă contextul RAG e suficient pentru a răspunde.

        Întreabă LLM-ul (prompt "rag_evaluate") dacă poate răspunde cu chunk-urile
        găsite; întoarce un OrchestratorFeedback (can_answer + ce lipsește/sugestie)
        care decide dacă mai facem o iterație de căutare sau trecem la răspuns.
        """
        logger.info(f"[EVALUATE] iter {state.iteration}")

        rag = state.rag_result
        if not rag or not rag.results:
            return {
                "feedback": OrchestratorFeedback(
                    can_answer=False,
                    missing_info="Nu s-au găsit documente relevante.",
                    suggestion="Reformulează query-ul sau scade threshold-ul.",
                )
            }

        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=self._build_context(state),
            max_score=round(rag.max_score, 3),
            avg_score=round(rag.avg_score, 3),
        )
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        try:
            feedback = OrchestratorFeedback.model_validate_json(extract_json(response))
        except Exception as e:
            # Dacă nu putem parsa evaluarea, mai bine răspundem cu ce avem decât
            # să intrăm în buclă (oricum _should_continue plafonează la max_iterations).
            logger.warning(f"[EVALUATE] parse eșuat ({e}) → presupun can_answer=True")
            feedback = OrchestratorFeedback(can_answer=True)

        logger.info(f"[EVALUATE] can_answer={feedback.can_answer}")
        return {"feedback": feedback}

    def node_answer(self, state: OrchestratorState) -> dict:
        """
        Generează răspunsul final din contextul RAG (prompt "rag_answer").

        status:
          - "success"  dacă orchestratorul a confirmat can_answer
          - "partial"  dacă răspundem fără confirmare (ex: am atins max_iterations)
          - "failed"   dacă nu avem deloc rezultate
        """
        logger.info("[ANSWER]")

        rag = state.rag_result
        if not rag or not rag.results:
            return {
                "answer": "Nu am găsit informații relevante în documente pentru a răspunde.",
                "status": "failed",
            }

        prompt = self.prompts.render(
            "rag_answer",
            query=state.query,
            context=self._build_context(state),
        )
        answer = self.llm.generate_sync([{"role": "user", "content": prompt}])

        can_answer = bool(state.feedback and state.feedback.can_answer)
        status = "success" if can_answer else "partial"

        logger.info(f"[ANSWER] status={status} ({len(answer)} caractere)")
        return {"answer": answer.strip(), "status": status}

    # === ROUTING ===

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        """Decide dacă continuăm căutarea sau răspundem."""
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.config.max_iterations:
            logger.info(f"[ROUTING] Max iterations ({self.config.max_iterations}) reached")
            return "answer"
        return "call_rag"

    # === GRAPH ===

    def build_graph(self):
        """Construiește graful Orchestrator."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)

        graph.add_edge(START, "call_rag")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"call_rag": "call_rag", "answer": "answer"}
        )
        graph.add_edge("answer", END)

        return graph.compile()
