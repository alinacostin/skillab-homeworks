"""QAAgent — orchestrare LLM + tools + prompts, cu optimizările din Tema 4.

Flux complet:
  0. intent    → routing ieftin cu sklearn, fallback la LLM sub prag
  1. memory    → LOAD istoricul sesiunii din PostgreSQL
  2. planner   → system prompt al loop-ului ReAct (cache_control: ephemeral pe Anthropic)
  3. ReAct     → Think / Act / Observe în paralel pentru tool calls multiple
  4. analyst   → critică & rafinează raw_answer
  5. summary   → format final user-facing
  6. extract   → JSON structurat (rulează doar pentru intent „extract")
  7. memory    → SAVE turul (user + assistant) în PostgreSQL
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

from memory import PersistentMemory
from prompts import get_prompt_registry
from tools import TOOL_REGISTRY, ToolWrapper

# Intent classifier — opțional: dacă sklearn/joblib lipsesc, agentul merge fără.
try:
    from intent import detect_intent, is_trained
except Exception:  # pragma: no cover - intent e o optimizare opțională
    detect_intent = None

    def is_trained() -> bool:
        return False

load_dotenv()

# Etichetele de intent + promptul de fallback LLM 
INTENT_LABELS = ["search", "extract", "summarize"]
_INTENT_SYSTEM = (
    "Ești un clasificator de intent. Clasifică query-ul în EXACT una dintre:\n"
    "- search: caută/găsește/arată documente sau informații\n"
    "- extract: extrage date structurate (sume, date, TVA, IBAN, părți)\n"
    "- summarize: rezumă/sintetizează/overview\n"
    "Răspunde DOAR cu eticheta (un cuvânt)."
)


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
    def __init__(
        self,
        provider: str | None = None,
        max_iterations: int = 8,
        *,
        use_memory: bool = False,
        memory: PersistentMemory | None = None,
        window: int | None = None,
        use_intent: bool | None = None,
        use_cache: bool | None = None,
        confidence_threshold: float | None = None,
    ):
        self.max_iterations = max_iterations
        self.prompts = get_prompt_registry()
        self.provider = (provider or os.getenv("DEFAULT_PROVIDER", "gemini")).lower()

        llm = LLMFactory.create(provider=provider)
        catalog = ToolWrapper.catalog()
        self.llm_with_tools = llm.bind_tools(catalog)
        self.llm_plain = llm  # pentru intent-fallback / analyst / summary / extract

        # ── Conversation Memory ──
        if memory is not None:
            self.memory: PersistentMemory | None = memory
        elif use_memory:
            win = window if window is not None else int(os.getenv("MEMORY_WINDOW", "10"))
            self.memory = PersistentMemory(window=win)
        else:
            self.memory = None

        # ── Prompt caching — doar pe Anthropic (cache_control: ephemeral) ──
        if use_cache is None:
            env = os.getenv("PROMPT_CACHE", "").strip().lower()
            use_cache = env in ("1", "true", "yes") if env else (self.provider == "anthropic")
        self.use_cache = bool(use_cache) and self.provider == "anthropic"

        # ── Intent classifier ──
        self.use_intent = is_trained() if use_intent is None else use_intent
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        )

        # usage cumulat din loop-ul ReAct (pentru măsurarea caching-ului)
        self._last_usage: dict[str, int] = {}

    # ---------- helpers ----------
    def _system_message(self) -> SystemMessage:
        """System prompt al loop-ului ReAct. Pe Anthropic îl marcăm cacheabil
        (cache_control: ephemeral) → prefix static servit din cache la 0.1x."""
        system_prompt = self.prompts.render(
            "planner",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            tools_description=_tools_description(),
        )
        if self.use_cache:
            return SystemMessage(content=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }])
        return SystemMessage(content=system_prompt)

    @staticmethod
    def _history_to_messages(history: list[dict] | None) -> list:
        """Mapează istoricul din DB la mesaje LangChain (user→Human, assistant→AI)."""
        msgs = []
        for m in history or []:
            if m.get("role") == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        return msgs

    def _accumulate_usage(self, resp: AIMessage) -> None:
        """Adună tokenii din usage_metadata (inclusiv cache_read / cache_creation)."""
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return
        u = self._last_usage
        u["input_tokens"] = u.get("input_tokens", 0) + (um.get("input_tokens", 0) or 0)
        u["output_tokens"] = u.get("output_tokens", 0) + (um.get("output_tokens", 0) or 0)
        details = um.get("input_token_details", {}) or {}
        u["cache_read"] = u.get("cache_read", 0) + (details.get("cache_read", 0) or 0)
        u["cache_creation"] = u.get("cache_creation", 0) + (details.get("cache_creation", 0) or 0)

    # ---------- intent ----------
    def _detect_intent_llm(self, question: str) -> str:
        """Fallback: clasificare intent cu LLM (varianta scumpă)."""
        resp = self.llm_plain.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        text = _content_to_text(resp.content).strip().lower()
        for lbl in INTENT_LABELS:
            if lbl in text:
                return lbl
        return "search"

    def _classify_intent(self, question: str) -> tuple[str | None, float, str]:
        """(label, confidence, via). sklearn întâi; sub prag → fallback LLM."""
        if not self.use_intent or detect_intent is None:
            return None, 0.0, "disabled"
        try:
            label, conf = detect_intent(question)
        except FileNotFoundError:
            return None, 0.0, "untrained"
        if conf < self.confidence_threshold:
            return self._detect_intent_llm(question), conf, "llm_fallback"
        return label, conf, "classifier"

    # ---------- ReAct loop ----------
    def _react_loop(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        messages = [self._system_message()]
        messages.extend(self._history_to_messages(history))   # context din memorie
        messages.append(HumanMessage(content=question))
        trace: list[dict[str, Any]] = []

        for i in range(1, self.max_iterations + 1):
            raspuns: AIMessage = self.llm_with_tools.invoke(messages)
            self._accumulate_usage(raspuns)
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

    # ---------- pipeline pași 4-6 ----------
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
    def ask(
        self, question: str, session_id: str | None = None, user_id: str = "default"
    ) -> dict[str, Any]:
        self._last_usage = {}

        # 0. INTENT — routing ieftin (sklearn) cu fallback LLM sub prag
        intent, intent_conf, intent_via = self._classify_intent(question)

        # 1. LOAD memorie — istoricul sesiunii din PostgreSQL
        history: list[dict] = []
        if self.memory and session_id:
            history = self.memory.load_messages(session_id)

        # 2-3. ReAct cu istoric injectat + system prompt cacheabil
        raw_answer, trace = self._react_loop(question, history=history)

        # 4. analyst
        refined = self._refine(
            "analyst", user_question=question, raw_answer=raw_answer, trace=trace
        )

        # 5. summary (răspunsul user-facing)
        summary = self._refine(
            "summary", user_question=question, refined_answer=refined
        )

        # 6. extract — JSON structurat doar pentru intent „extract"
        #    (sau când intent e dezactivat → comportamentul clasic, mereu extract).
        if intent in (None, "extract"):
            extract_text = self._refine("extract", summary=summary, trace=trace)
            structured = self._parse_json(extract_text)
        else:
            structured = {}

        # 7. SAVE memorie — persistăm turul (user + assistant) atomic
        if self.memory and session_id:
            self.memory.save_turn(session_id, question, summary, user_id=user_id)

        return {
            "answer": summary,
            "structured": structured,
            "trace": trace,
            "raw_answer": raw_answer,
            "refined_answer": refined,
            "intent": intent,
            "intent_confidence": intent_conf,
            "intent_via": intent_via,
            "usage": dict(self._last_usage),
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
    # SESSION_ID în env → conversația persistă în PostgreSQL între rulări.
    session_id = os.getenv("SESSION_ID")
    print(f"\n[provider: {provider}] [session: {session_id or '—'}]\n", file=sys.stderr)

    agent = QAAgent(use_memory=bool(session_id))
    rezultat = agent.ask(question, session_id=session_id)

    print(rezultat["answer"])

    if rezultat.get("intent"):
        print(
            f"\n[intent: {rezultat['intent']} "
            f"(conf={rezultat['intent_confidence']:.2f}, via {rezultat['intent_via']})]",
            file=sys.stderr,
        )
    u = rezultat.get("usage") or {}
    if u.get("cache_read") or u.get("cache_creation"):
        print(
            f"[cache: read={u.get('cache_read', 0)} "
            f"creation={u.get('cache_creation', 0)} input={u.get('input_tokens', 0)}]",
            file=sys.stderr,
        )

    print("\n--- Structured ---")
    print(json.dumps(rezultat["structured"], ensure_ascii=False, indent=2))
    print("\n--- Trace ---")
    for step in rezultat["trace"]:
        print(f"  [{step['iteration']}] {step['tool']}({step['args']}) → {step['result'][:120]}")


if __name__ == "__main__":
    main()
