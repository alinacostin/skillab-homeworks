"""
Analyst Agent - TODO: Implementează nodurile
"""
import json
import logging
from pathlib import Path

import pandas as pd
from langgraph.graph import StateGraph, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry
from skillab.tools import ToolWrapper, TOOL_REGISTRY

from json_utils import extract_json
from state import AnalystState, QueryStep, ToolStep, StepResult
from nl2sql_agent import NL2SQLAgent

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class AnalystAgent:
    """
    Flow:
        make_plan → execute_step (loop) → synthesize

    Plan steps:
        - QueryStep: query pe tabel via NL2SQL
        - ToolStep: apel tool din registry (join_data, filter_data)

    Rezultatele sunt stocate în state.slices[step.id] = DataFrame
    """

    def __init__(
        self,
        tables_config: dict[str, dict],
        llm: LLMProvider | None = None,
    ):
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load table info + create NL2SQL agents
        self.tables_info = []
        self.sql_agents: dict[str, NL2SQLAgent] = {}

        for table_name, config in tables_config.items():
            schema = json.loads(Path(config["schema_path"]).read_text())
            self.tables_info.append({
                "name": table_name,
                "description": schema.get("description", ""),
                "columns": list(schema.get("columns", {}).keys()),
            })
            self.sql_agents[table_name] = NL2SQLAgent(
                table_name=table_name,
                schema_path=config["schema_path"],
                llm=self.llm,
                business_path=config.get("business_path"),
            )

        # Tool catalog pentru prompt — DOAR tool-urile pe DataFrame (join/filter).
        # Registry-ul conține și tool-urile de pre/postprocess (detect_intent etc.),
        # dar planificatorul Analyst nu trebuie să le vadă (lucrează pe slices/DataFrames).
        df_tools = {n for n, t in TOOL_REGISTRY.items() if "input_dfs" in t["params_model"].model_fields}
        self.tools_catalog = "Tools disponibile:\n" + "\n".join(
            f"- {info.name}: {info.description}"
            for info in ToolWrapper.catalog() if info.name in df_tools
        )

        self.graph = self._build_graph()

    # === NODES ===

    def node_make_plan(self, state: AnalystState) -> dict:
        """
        TODO: Creează plan.

        1. Renderează prompt "analyst_plan" cu tables_info și tools_catalog
        2. Apelează LLM
        3. Parsează JSON
        4. Return {"reasoning": ..., "plan": [...], "current_step": 0}

        Hint pentru prompt:
            - tables_info: lista de tabele disponibile
            - tools_catalog: tools disponibile (join_data, filter_data)

        Plan format:
            [
                {"id": "q1", "action": "query", "table": "achizitii", "sub_question": "..."},
                {"id": "q2", "action": "query", "table": "anunturi", "sub_question": "..."},
                {"id": "joined", "action": "tool", "tool_name": "join_data", "input_steps": ["q1", "q2"], "params": {...}}
            ]
        """
        logger.info(f"[PLAN] {state.question}")

        prompt = self.prompts.render(
            "analyst_plan",
            tables=self.tables_info,
            tools_catalog=self.tools_catalog,
            question=state.question,
            history=[],
        )
        response = self.llm.generate_sync([{"role": "user", "content": prompt}])

        try:
            data = json.loads(extract_json(response))
        except Exception as e:
            logger.warning(f"[PLAN] parse JSON eșuat: {e}")
            return {"reasoning": "", "plan": [], "current_step": 0, "slices": {}, "step_results": []}

        reasoning = data.get("reasoning", "")
        plan: list = []
        for raw in data.get("steps", []):
            action = raw.get("action")
            try:
                if action == "query":
                    plan.append(QueryStep(**raw))
                elif action == "tool":
                    plan.append(ToolStep(**raw))
                else:
                    logger.warning(f"[PLAN] action necunoscut '{action}' — pas ignorat")
            except Exception as e:
                logger.warning(f"[PLAN] pas invalid {raw}: {e}")

        logger.info(f"[PLAN] {len(plan)} pași: {[s.id for s in plan]}")
        return {
            "reasoning": reasoning,
            "plan": plan,
            "current_step": 0,
            "slices": {},
            "step_results": [],
        }

    def node_execute_step(self, state: AnalystState) -> dict:
        """Execută pasul curent din plan și stochează rezultatul în slices."""
        step_idx = state.current_step
        step = state.plan[step_idx]
        logger.info(f"[EXECUTE] step {step.id}: {step.action}")

        if step.action == "query":
            result, df = self._execute_query(step)
        elif step.action == "tool":
            result, df = self._execute_tool(step, state.slices)
        else:
            result = StepResult(
                step_id=step.id,
                action=step.action,
                description=f"Unknown action: {step.action}",
                status="failed",
                error=f"Action '{step.action}' not supported",
            )
            df = None

        # Update slices dacă execuția a reușit
        new_slices = dict(state.slices)
        if df is not None:
            new_slices[step.id] = df

        return {
            "slices": new_slices,
            "step_results": state.step_results + [result],
            "current_step": step_idx + 1,
        }

    def _execute_query(self, step: QueryStep) -> tuple[StepResult, pd.DataFrame | None]:
        """Execută query pe tabel via NL2SQL agent."""
        agent = self.sql_agents.get(step.table)
        if not agent:
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=f"No agent for table '{step.table}'",
            ), None

        try:
            nl2sql_result = agent.run(step.sub_question)
        except Exception as e:
            logger.exception(f"NL2SQL failed for step {step.id}")
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=str(e),
            ), None

        if nl2sql_result.status != "success":
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=nl2sql_result.execution_error or nl2sql_result.validation_error,
            ), None

        return StepResult(
            step_id=step.id,
            action=step.action,
            description=step.sub_question,
            status="success",
            row_count=len(nl2sql_result.result),
        ), nl2sql_result.result

    def _execute_tool(
        self,
        step: ToolStep,
        slices: dict[str, pd.DataFrame],
    ) -> tuple[StepResult, pd.DataFrame | None]:
        """Execută un tool din registry folosind slices."""
        # Validare input_steps - verifică că ID-urile există în slices
        for input_id in step.input_steps:
            if input_id not in slices:
                return StepResult(
                    step_id=step.id,
                    action=step.action,
                    description=f"{step.tool_name}",
                    status="failed",
                    error=f"Input step '{input_id}' not found in slices",
                ), None

        # Construiește argumentele pentru tool (convenție: input_dfs + params)
        input_dfs = [slices[input_id] for input_id in step.input_steps]
        args = {"input_dfs": input_dfs, **step.params}

        # Execută tool
        try:
            result_df = ToolWrapper.call(step.tool_name, args)
        except NotImplementedError as e:
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=f"{step.tool_name}",
                status="failed",
                error=f"Tool not implemented: {e}",
            ), None
        except Exception as e:
            logger.exception(f"Tool {step.tool_name} failed")
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=f"{step.tool_name}",
                status="failed",
                error=str(e),
            ), None

        return StepResult(
            step_id=step.id,
            action=step.action,
            description=f"{step.tool_name}({step.params})",
            status="success",
            row_count=len(result_df),
        ), result_df

    def node_synthesize(self, state: AnalystState) -> dict:
        """
        TODO: Sintetizează răspunsul.

        1. Renderează prompt "analyst_synthesize" cu results
        2. Apelează LLM
        3. Return {"answer": ..., "status": "success" | "failed"}

        Hint: ultimul slice (sau cel specificat) conține rezultatul final
            final_df = state.slices[state.plan[-1].id]
        """
        logger.info("[SYNTHESIZE]")

        if not state.plan:
            return {
                "answer": "Nu am putut construi un plan pentru această întrebare.",
                "status": "no_plan",
            }

        # rezultatul final = slice-ul ultimului pas din plan (dacă există)
        final_df = state.slices.get(state.plan[-1].id)
        final_data = ""
        if final_df is not None and len(final_df):
            final_data = final_df.head(10).to_string(index=False)

        prompt = self.prompts.render(
            "analyst_synthesize",
            question=state.question,
            reasoning=state.reasoning,
            results=state.step_results,
            final_data=final_data,
        )
        answer = self.llm.generate_sync([{"role": "user", "content": prompt}])

        any_success = any(r.status == "success" for r in state.step_results)
        status = "success" if any_success else "failed"
        logger.info(f"[SYNTHESIZE] status={status}")
        return {"answer": answer.strip(), "status": status}

    # === ROUTING ===

    def _route_after_plan(self, state: AnalystState) -> str:
        return "execute_step" if state.plan else "synthesize"

    def _route_after_execute(self, state: AnalystState) -> str:
        return "execute_step" if state.current_step < len(state.plan) else "synthesize"

    # === GRAPH ===

    def _build_graph(self):
        graph = StateGraph(AnalystState)

        graph.add_node("make_plan", self.node_make_plan)
        graph.add_node("execute_step", self.node_execute_step)
        graph.add_node("synthesize", self.node_synthesize)

        graph.set_entry_point("make_plan")
        graph.add_conditional_edges("make_plan", self._route_after_plan, ["execute_step", "synthesize"])
        graph.add_conditional_edges("execute_step", self._route_after_execute, ["execute_step", "synthesize"])
        graph.add_edge("synthesize", END)

        return graph.compile()

    def chat(self, question: str) -> AnalystState:
        """Execută agentul."""
        initial = AnalystState(question=question)
        return self.graph.invoke(initial)
