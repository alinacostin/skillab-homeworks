"""Benchmark Intent Detection: LLM vs sklearn.

Compară, pe același set held-out etichetat, trei metrici între un apel LLM de
clasificare și classifier-ul TF-IDF + LogisticRegression antrenat:

    • Latență   — time.perf_counter
    • Cost      — din token usage real (Anthropic) sau cifre de referință
    • Accuracy  — predicții vs etichete gold

Rulează (din proiect/):
    python benchmark_intent.py

Notă: partea sklearn rulează mereu (local, offline). Partea LLM cere un provider
configurat (.env: DEFAULT_PROVIDER + cheia respectivă); fără el e sărită automat.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agent import LLMFactory, _content_to_text
from intent import detect_intent, is_trained
from intent.intent_data import LABELS, TEST_DATA

load_dotenv()

# Preț de referință pentru estimarea costului LLM (Anthropic Sonnet, $/1M tokeni).
# Suprascrie din env dacă folosești alt model/provider.
PRICE_IN_PER_MTOK = float(os.getenv("LLM_PRICE_IN_PER_MTOK", "3.0"))
PRICE_OUT_PER_MTOK = float(os.getenv("LLM_PRICE_OUT_PER_MTOK", "15.0"))
# Fallback dacă usage-ul nu e disponibil: cifră de referință (~$20/1000 calls).
LLM_FALLBACK_COST_PER_1K = float(os.getenv("LLM_FALLBACK_COST_PER_1K", "20.0"))

DAILY_VOLUME = int(os.getenv("BENCH_DAILY_VOLUME", "10000"))  # scenariu de extrapolare

_INTENT_SYSTEM = (
    "Ești un clasificator de intent. Clasifică query-ul utilizatorului în EXACT una "
    "dintre categoriile:\n"
    "- search: caută/găsește/arată documente sau informații\n"
    "- extract: extrage date structurate (sume, date, TVA, IBAN, părți, numere)\n"
    "- summarize: rezumă/sintetizează/overview\n"
    "Răspunde DOAR cu eticheta (un singur cuvânt), fără explicații."
)


def _parse_label(text: str) -> str:
    """Normalizează răspunsul LLM la una dintre etichetele cunoscute."""
    t = text.strip().lower()
    for lbl in LABELS:
        if lbl in t:
            return lbl
    return t.split()[0] if t else "?"


def detect_intent_llm(llm, query: str) -> tuple[str, dict]:
    """Clasificare intent cu LLM (varianta SCUMPĂ). Întoarce (label, usage)."""
    resp = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=query),
    ])
    label = _parse_label(_content_to_text(resp.content))
    usage = getattr(resp, "usage_metadata", None) or {}
    return label, usage


def _bench_sklearn() -> dict:
    # warm-up: cold start (încărcarea modelului) măsurat separat
    t0 = time.perf_counter()
    detect_intent(TEST_DATA[0][0])
    cold_start_ms = (time.perf_counter() - t0) * 1000

    correct, times = 0, []
    for query, gold in TEST_DATA:
        t0 = time.perf_counter()
        label, _conf = detect_intent(query)
        times.append(time.perf_counter() - t0)
        correct += int(label == gold)

    return {
        "name": "sklearn",
        "avg_ms": sum(times) / len(times) * 1000,
        "accuracy": correct / len(TEST_DATA),
        "cost_per_1k": 0.01,  # local, ~$0 (cifră simbolică de referință)
        "offline": True,
        "cold_start_ms": cold_start_ms,
    }


def _bench_llm() -> dict | None:
    try:
        llm = LLMFactory.create(temperature=0)
    except Exception as e:  # noqa: BLE001
        print(f"[LLM sărit] nu pot crea provider-ul: {type(e).__name__}: {e}")
        return None

    correct, times = 0, []
    in_tok = out_tok = 0
    have_usage = False
    try:
        for query, gold in TEST_DATA:
            t0 = time.perf_counter()
            label, usage = detect_intent_llm(llm, query)
            times.append(time.perf_counter() - t0)
            correct += int(label == gold)
            if usage:
                have_usage = True
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
    except Exception as e:  # noqa: BLE001 — quota/rețea/cheie lipsă
        print(f"[LLM sărit] apel eșuat: {type(e).__name__}: {str(e)[:160]}")
        return None

    n = len(TEST_DATA)
    if have_usage and (in_tok or out_tok):
        cost_total = (in_tok * PRICE_IN_PER_MTOK + out_tok * PRICE_OUT_PER_MTOK) / 1e6
        cost_per_1k = cost_total / n * 1000
    else:
        cost_per_1k = LLM_FALLBACK_COST_PER_1K

    return {
        "name": f"LLM ({llm.__class__.__name__})",
        "avg_ms": sum(times) / len(times) * 1000,
        "accuracy": correct / n,
        "cost_per_1k": cost_per_1k,
        "offline": False,
        "tokens": (in_tok, out_tok) if have_usage else None,
    }


def _print_table(llm_stats: dict | None, ml_stats: dict) -> None:
    print("\n" + "=" * 64)
    print("COMPARAȚIE: LLM vs sklearn  (intent detection)")
    print("=" * 64)
    header = f"{'Metric':<26}{'LLM':>18}{'sklearn':>18}"
    print(header)
    print("-" * 64)

    def row(label, llm_val, ml_val):
        lv = llm_val if llm_stats else "—"
        print(f"{label:<26}{str(lv):>18}{str(ml_val):>18}")

    row("Cost / 1000 calls",
        f"${llm_stats['cost_per_1k']:.2f}" if llm_stats else "—",
        f"${ml_stats['cost_per_1k']:.2f}")
    row("Latență medie",
        f"{llm_stats['avg_ms']:.0f}ms" if llm_stats else "—",
        f"{ml_stats['avg_ms']:.2f}ms")
    row("Accuracy (held-out)",
        f"{llm_stats['accuracy']:.0%}" if llm_stats else "—",
        f"{ml_stats['accuracy']:.0%}")
    row("Offline", "❌" if llm_stats else "—", "✅")
    print("-" * 64)
    print(f"Cold start sklearn (load model): {ml_stats['cold_start_ms']:.0f}ms")

    if llm_stats:
        speedup = llm_stats["avg_ms"] / ml_stats["avg_ms"] if ml_stats["avg_ms"] else 0
        cost_ratio = llm_stats["cost_per_1k"] / ml_stats["cost_per_1k"] if ml_stats["cost_per_1k"] else 0
        print(f"\n⚡ sklearn ~{speedup:.0f}× mai rapid   💰 ~{cost_ratio:.0f}× mai ieftin")
        # Extrapolare la scară
        llm_day = llm_stats["cost_per_1k"] / 1000 * DAILY_VOLUME
        ml_day = ml_stats["cost_per_1k"] / 1000 * DAILY_VOLUME
        print(f"\nLa {DAILY_VOLUME:,} req/zi:  LLM ${llm_day:,.0f}/zi (${llm_day*30:,.0f}/lună)  "
              f"vs sklearn ${ml_day:,.2f}/zi")
        print(f"💡 Economie estimată: ${(llm_day - ml_day) * 30:,.0f}/lună")
        if llm_stats.get("tokens"):
            print(f"   (cost LLM din usage real: {llm_stats['tokens'][0]} in / "
                  f"{llm_stats['tokens'][1]} out tokeni pe {len(TEST_DATA)} apeluri)")


def main() -> None:
    if not is_trained():
        print("Model neantrenat. Rulează mai întâi: python -m intent.train_intent")
        return

    print(f"Set de test: {len(TEST_DATA)} exemple held-out, clase: {LABELS}")
    ml_stats = _bench_sklearn()
    llm_stats = _bench_llm()
    _print_table(llm_stats, ml_stats)


if __name__ == "__main__":
    main()
