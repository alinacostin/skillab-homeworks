"""demo.py — demonstrație end-to-end

    1. ingest documente (facturi + contracte) → pgvector — DOAR dacă baza e goală
    2. întrebări puse agentului, care folosește tool-ul `search_documents`

Rulare: python demo.py   (necesită docker-compose up + alembic upgrade head)
"""

from dotenv import load_dotenv

from agent import QAAgent
from database import transaction
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


def main() -> None:
    ensure_ingested()

    print("\n== Întrebări către agent (folosește tool-ul search_documents) ==")
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


if __name__ == "__main__":
    main()
