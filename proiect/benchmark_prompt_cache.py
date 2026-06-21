"""Benchmark Prompt Caching (Anthropic) — Tema 4 · Part 2.

Activează prompt caching cu `cache_control: ephemeral` pe un prefix STATIC mare
(system + document fix) și MĂSOARĂ economia: tokeni serviți din cache + reducerea
de latență (apel 1 = cache MISS / write, apel 2 = cache HIT / read la 0.1x).

Două părți:
  A · SDK nativ Anthropic        — apel direct prin SDK.
  B · LangChain ChatAnthropic    — mecanismul folosit efectiv de QAAgent (același
                                   `cache_control` pe SystemMessage, usage din usage_metadata).

⚠️ Prag minim de caching: 1024 tokeni (Claude 3.x) / 2048 (Sonnet 4.x). Sub prag,
   cache-ul NU se activează (cache_creation rămâne 0, fără eroare). Documentul de mai
   jos e dimensionat să treacă pragul. Cere ANTHROPIC_API_KEY.

Rulează (din proiect/):
    DEFAULT_PROVIDER=anthropic python benchmark_prompt_cache.py
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-5-20250929"
QUESTIONS = ["Care e penalizarea pentru întârziere?", "Cât e TVA-ul aplicat?"]


def big_document() -> str:
    """Prefix static mare (> 2048 tokeni). În realitate = un contract/document."""
    para = (
        "Contract de prestări servicii încheiat între Beneficiar și Furnizor. "
        "Termenii includ: livrare în 30 de zile calendaristice, penalizare de 0.1% pe "
        "zi de întârziere din valoarea contractului, TVA 19% aplicat la valoarea netă, "
        "plata în 15 zile de la data facturii. Sunt aplicabile clauze de "
        "confidențialitate, garanție de 24 de luni și forță majoră. "
    )
    return para * 60  # repetăm ca să depășim pragul fără să inventăm 5 pagini


def _report(dt1: float, dt2: float, cache_read: int, fresh: int) -> None:
    if cache_read:
        # cache_read costă 0.1x față de input normal → economie pe partea cache-uită
        full = cache_read + fresh
        cached_cost = cache_read * 0.1 + fresh
        saved = (1 - cached_cost / full) * 100 if full else 0
        print(f"\n💰 {cache_read} tokeni serviți din cache (0.1x) → ~{saved:.0f}% "
              f"reducere pe inputul apelului 2")
    else:
        print("\n⚠️ cache_read=0 — prefixul a fost sub prag (2048 tok pe Sonnet 4.x) "
              "sau TTL-ul cache-ului (5 min) a expirat.")
    if dt1 > 0:
        print(f"⏱  Latență: apel 1 {dt1*1000:.0f}ms → apel 2 {dt2*1000:.0f}ms "
              f"(reducere {(1 - dt2/dt1)*100:.0f}%)")


def demo_native() -> None:
    print("=" * 64)
    print("PARTEA A · Prompt Caching — SDK nativ Anthropic")
    print("=" * 64)
    try:
        import anthropic
    except ImportError:
        print("anthropic lipsește (pip install anthropic) — sărit.")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY lipsește — sărit.")
        return

    client = anthropic.Anthropic()
    system_blocks = [
        {"type": "text", "text": "Ești un analist de contracte."},
        {"type": "text", "text": big_document(),
         "cache_control": {"type": "ephemeral"}},  # ← prefixul cache-uit
    ]

    def ask(q: str):
        t0 = time.perf_counter()
        r = client.messages.create(
            model=MODEL, max_tokens=256, system=system_blocks,
            messages=[{"role": "user", "content": q}],
        )
        return time.perf_counter() - t0, r.usage

    try:
        dt1, u1 = ask(QUESTIONS[0])
        print(f"\n→ Apel 1 (MISS): {dt1*1000:.0f}ms  "
              f"creation={u1.cache_creation_input_tokens} "
              f"read={u1.cache_read_input_tokens} fresh={u1.input_tokens}")
        dt2, u2 = ask(QUESTIONS[1])
        print(f"→ Apel 2 (HIT):  {dt2*1000:.0f}ms  "
              f"creation={u2.cache_creation_input_tokens} "
              f"read={u2.cache_read_input_tokens} fresh={u2.input_tokens}")
        _report(dt1, dt2, u2.cache_read_input_tokens, u2.input_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"[eroare apel Anthropic] {type(e).__name__}: {str(e)[:160]}")


def demo_langchain() -> None:
    print("\n" + "=" * 64)
    print("PARTEA B · Prompt Caching — calea LangChain folosită de QAAgent")
    print("=" * 64)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY lipsește — sărit.")
        return
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("langchain-anthropic lipsește — sărit.")
        return

    llm = ChatAnthropic(model=MODEL, temperature=0, max_tokens=256)
    # Exact ca în QAAgent._system_message: SystemMessage cu bloc cache_control.
    sys_msg = SystemMessage(content=[{
        "type": "text",
        "text": "Ești un analist de contracte.\n" + big_document(),
        "cache_control": {"type": "ephemeral"},
    }])

    def ask(q: str):
        t0 = time.perf_counter()
        r = llm.invoke([sys_msg, HumanMessage(content=q)])
        dt = time.perf_counter() - t0
        um = getattr(r, "usage_metadata", None) or {}
        d = um.get("input_token_details", {}) or {}
        return dt, um.get("input_tokens", 0), d.get("cache_creation", 0), d.get("cache_read", 0)

    try:
        dt1, in1, cc1, cr1 = ask(QUESTIONS[0])
        print(f"\n→ Apel 1 (MISS): {dt1*1000:.0f}ms  creation={cc1} read={cr1} input={in1}")
        dt2, in2, cc2, cr2 = ask(QUESTIONS[1])
        print(f"→ Apel 2 (HIT):  {dt2*1000:.0f}ms  creation={cc2} read={cr2} input={in2}")
        # LangChain: input_tokens = TOTAL (include cache); fresh = total - cache_read - cache_creation
        # (la SDK-ul nativ, input_tokens e deja doar partea fresh — vezi Partea A).
        _report(dt1, dt2, cr2, max(0, in2 - cr2 - cc2))
    except Exception as e:  # noqa: BLE001
        print(f"[eroare apel LangChain] {type(e).__name__}: {str(e)[:160]}")


def main() -> None:
    print(f"Model: {MODEL}\n")
    demo_native()
    demo_langchain()


if __name__ == "__main__":
    main()
