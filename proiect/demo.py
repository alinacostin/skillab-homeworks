"""demo.py — demonstrație end-to-end.

Subcomenzi:
    python demo.py            # RAG: ingest documente + întrebări către agent
    python demo.py rag        # idem
    python demo.py memory     # Tema 4: conversation memory care supraviețuiește „restart-ului"

Necesită: docker compose up -d  +  alembic upgrade head.
Pentru memory/caching pe Anthropic: DEFAULT_PROVIDER=anthropic (și DEFAULT_MODEL gol/Claude).
"""

import sys

from dotenv import load_dotenv

from agent import QAAgent
from database import transaction
from memory import PersistentMemory
from pipeline import DOCUMENTS_DIR, ingest
from repositories import DocumentRepository

load_dotenv()

INTREBARI = [
    "Ce clauze de reziliere avem în contracte?",  # întrebarea canonică din temă (hw4.pdf)
    "Ce clauze de confidențialitate avem în contracte?",
    "Care este termenul de plată al facturilor?",
    "Cine este prestatorul în contractul de consultanță și ce valoare are?",
    "Care este totalul de plată al facturii FV-2024-001?",
]


def ensure_ingested() -> None:
    """Ingestează doar dacă baza e goală — evită duplicate și apeluri LLM redundante."""
    with transaction() as db:
        already = DocumentRepository(db).get_all(limit=1)
    if already:
        print("== Documente deja în DB — sar peste ingest ==")
        return
    paths = sorted(DOCUMENTS_DIR.glob("*.txt"))
    print(f"== Ingest {len(paths)} documente ==")
    ingest(paths)


def demo_rag() -> None:
    ensure_ingested()

    print("\n== Întrebări către agent (search_documents + intent + cache) ==")
    agent = QAAgent()
    for q in INTREBARI:
        print(f"\nQ: {q}")
        try:
            rezultat = agent.ask(q)
        except Exception as e:
            # Rate limit / quota / network — nu omoram tot demo-ul pentru o întrebare.
            print(f"A: (apel LLM eșuat — probabil rate limit/quota) {type(e).__name__}: {str(e)[:160]}")
            continue
        print(f"A: {rezultat['answer']}")
        tools_used = [step["tool"] for step in rezultat["trace"]]
        if tools_used:
            print(f"   (tools: {', '.join(tools_used)})")
        if rezultat.get("intent"):
            print(f"   (intent: {rezultat['intent']} {rezultat['intent_confidence']:.2f} "
                  f"via {rezultat['intent_via']})")
        u = rezultat.get("usage") or {}
        if u.get("cache_read") or u.get("cache_creation"):
            print(f"   (cache: read={u.get('cache_read', 0)} creation={u.get('cache_creation', 0)})")


def demo_memory() -> None:
    """Tema 4 · Part 1 — conversation memory persistentă (supraviețuiește restart-ului)."""
    session_id = "demo-andrei"
    print("== Demo Conversation Memory (PostgreSQL) ==")
    PersistentMemory().clear(session_id)  # pornim de la zero

    # --- Sesiunea 1 ---
    print("\n--- Sesiunea 1 (agent #1) ---")
    agent1 = QAAgent(use_memory=True)
    q1 = "Mă numesc Andrei și mă interesează contractul cu DataPro."
    print(f"Q1: {q1}")
    try:
        print(f"A1: {agent1.ask(q1, session_id=session_id)['answer'][:200]}")
    except Exception as e:
        print(f"(apel LLM eșuat: {type(e).__name__}) — verifică providerul în .env")
        return

    # --- „RESTART": agent NOU, fără nimic în RAM, citește memoria din Postgres ---
    print("\n--- 🔌 RESTART: agent #2 (proces nou, RAM gol) ---")
    agent2 = QAAgent(use_memory=True)
    q2 = "Cum mă numesc și ce contract mă interesează?"
    print(f"Q2: {q2}")
    try:
        print(f"A2: {agent2.ask(q2, session_id=session_id)['answer'][:200]}")
    except Exception as e:
        print(f"(apel LLM eșuat: {type(e).__name__})")
        return

    hist = PersistentMemory().load_messages(session_id)
    print(f"\n💾 În PostgreSQL: {len(hist)} mesaje (memoria supraviețuiește restart-ului).")
    print("   → dacă A2 menționează 'Andrei' și 'DataPro', memoria funcționează.")


DEMOS = {"rag": demo_rag, "memory": demo_memory}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "rag"
    fn = DEMOS.get(which)
    if not fn:
        print(f"Demo necunoscut: {which}. Disponibile: {', '.join(DEMOS)}")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
