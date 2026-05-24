"""QAAgent — orchestrare LLM + tools + prompts.

Flux complet:
  1. planner   → system prompt al loop-ului ReAct
  2. ReAct     → Think / Act / Observe în paralel pentru tool calls multiple
  3. analyst   → critică & rafinează raw_answer
  4. summary   → format final user-facing
  5. extract   → JSON structurat {answer, key_facts, sources, tools_used}
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from prompts import get_prompt_registry
from tools import TOOL_REGISTRY, ToolWrapper

load_dotenv()


# Logica de instanțiere centralizată într-un singur loc.
class LLMFactory:
    @staticmethod
    def create(provider: str | None = None, **kwargs) -> BaseChatModel:
        provider = (provider or os.getenv("DEFAULT_PROVIDER", "gemini")).lower()
        kwargs.setdefault("temperature", 0.2)

        env_model = os.getenv("DEFAULT_MODEL")
        defaults = {
            "ollama": "llama3.2",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4-5-20250929",
        }
        model = kwargs.pop("model", env_model or defaults.get(provider))

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, **kwargs)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, **kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, **kwargs)

        raise ValueError(f"Provider necunoscut: {provider}")


def _tools_description() -> str:
    return "\n".join(
        f"- {name}: {entry['description']}"
        for name, entry in TOOL_REGISTRY.items()
    )


async def _execute_tool_async(tool_call: dict[str, Any]) -> ToolMessage:
    """Rulează un tool sincron într-un thread separat."""
    name = tool_call["name"]
    args = tool_call["args"]
    rezultat = await asyncio.to_thread(ToolWrapper.call, name, args)
    return ToolMessage(content=rezultat, tool_call_id=tool_call["id"])


async def _execute_all_tools(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Toate tool-urile cerute de LLM rulează în paralel."""
    tasks = [_execute_tool_async(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)


def _content_to_text(content: Any) -> str:
    """Normalizează AIMessage.content — Gemini returnează list de content parts,
    Ollama/Anthropic returnează string. Aducem mereu la string."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


class QAAgent:
    def __init__(self, provider: str | None = None, max_iterations: int = 8):
        self.max_iterations = max_iterations
        self.prompts = get_prompt_registry()

        llm = LLMFactory.create(provider=provider)
        catalog = ToolWrapper.catalog()
        self.llm_with_tools = llm.bind_tools(catalog)
        self.llm_plain = llm  # pentru analyst / summary / extract (fără tools)

    # ---------- ReAct loop ----------
    def _react_loop(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        system_prompt = self.prompts.render(
            "planner",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            tools_description=_tools_description(),
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        trace: list[dict[str, Any]] = []

        for i in range(1, self.max_iterations + 1):
            raspuns: AIMessage = self.llm_with_tools.invoke(messages)
            messages.append(raspuns)

            if not raspuns.tool_calls:
                return _content_to_text(raspuns.content).strip(), trace

            tool_messages = asyncio.run(_execute_all_tools(raspuns.tool_calls))
            for tc, tm in zip(raspuns.tool_calls, tool_messages):
                trace.append({
                    "iteration": i,
                    "tool": tc["name"],
                    "args": tc["args"],
                    "result": tm.content,
                })
                messages.append(tm)

        return (
            f"(Limită atinsă după {self.max_iterations} iterații fără răspuns final.)",
            trace,
        )

    # ---------- pipeline pași 3-5 ----------
    def _refine(self, prompt_name: str, **vars) -> str:
        prompt = self.prompts.render(prompt_name, **vars)
        raspuns = self.llm_plain.invoke([HumanMessage(content=prompt)])
        return _content_to_text(raspuns.content).strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    # ---------- public ----------
    def ask(self, question: str) -> dict[str, Any]:
        raw_answer, trace = self._react_loop(question)

        refined = self._refine(
            "analyst",
            user_question=question,
            raw_answer=raw_answer,
            trace=trace,
        )

        summary = self._refine(
            "summary",
            user_question=question,
            refined_answer=refined,
        )

        extract_text = self._refine(
            "extract",
            summary=summary,
            trace=trace,
        )

        return {
            "answer": summary,
            "structured": self._parse_json(extract_text),
            "trace": trace,
            "raw_answer": raw_answer,
            "refined_answer": refined,
        }


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Întrebare: ", end="", flush=True)
        question = sys.stdin.readline().strip()
        if not question:
            print("Niciun input.", file=sys.stderr)
            sys.exit(1)

    provider = os.getenv("DEFAULT_PROVIDER", "gemini")
    print(f"\n[provider: {provider}]\n", file=sys.stderr)

    agent = QAAgent()
    rezultat = agent.ask(question)

    print(rezultat["answer"])
    print("\n--- Structured ---")
    print(json.dumps(rezultat["structured"], ensure_ascii=False, indent=2))
    print("\n--- Trace ---")
    for step in rezultat["trace"]:
        print(f"  [{step['iteration']}] {step['tool']}({step['args']}) → {step['result'][:120]}")


if __name__ == "__main__":
    main()
