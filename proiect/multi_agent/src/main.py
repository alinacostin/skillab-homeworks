"""
Main — demo end-to-end pentru sistemul multi-agent (Tema 3).

Rulează:
  1. Data Reader        — un singur graf cu rag/csv/sql + retry/fallback
  2. Orchestrator + RAG — supervizor cu buclă evaluate → refine
  3. Analyst + NL2SQL   — plan → query → join/filter → synthesize
  4. Supervisor         — detect_intent → alege workeri → aggregate

Config LLM din .env (LLM_PROVIDER, *_API_KEY, *_MODEL). Fiecare demo e
protejat de try/except, ca o eroare de quota la unul să nu oprească restul.

Rulare:  cd src && python main.py        (sau: python main.py datareader|rag|sql|supervisor)
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillab-py" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from skillab import get_llm
from state import OrchestratorState

DATA_DIR = Path(__file__).parent.parent / "data"
NL2SQL_DIR = DATA_DIR / "nl2sql_agent"

# Alias-uri provider (gemini -> google, ollama -> local) + env var de model
_PROVIDER_ALIASES = {"gemini": "google", "ollama": "local"}
_MODEL_ENV_VARS = {
    "google": "GOOGLE_MODEL", "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL", "local": "OLLAMA_MODEL",
}


def _resolve_provider(provider: str | None) -> str | None:
    return _PROVIDER_ALIASES.get(provider.lower(), provider.lower()) if provider else None


def _get_model_from_env(provider: str | None) -> str | None:
    resolved = _resolve_provider(provider)
    if not resolved:
        return None
    return os.getenv("LLM_MODEL") or os.getenv(_MODEL_ENV_VARS.get(resolved, f"{resolved.upper()}_MODEL"))


LLM_PROVIDER = _resolve_provider(os.getenv("LLM_PROVIDER"))
LLM_MODEL = _get_model_from_env(os.getenv("LLM_PROVIDER"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
for _n in ("httpx", "sentence_transformers", "transformers", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def _make_llm():
    llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    print(f"LLM: {llm.name} / {llm.model}")
    return llm


def _tables_config() -> dict:
    return {
        "achizitii_directe": {
            "schema_path": str(NL2SQL_DIR / "schema_achizitii_directe.json"),
            "business_path": str(NL2SQL_DIR / "business_achizitii_directe.json"),
        },
        "anunturi_initiere": {
            "schema_path": str(NL2SQL_DIR / "schema_anunturi_initiere.json"),
            "business_path": str(NL2SQL_DIR / "business_anunturi_initiere.json"),
        },
    }


def demo_data_reader(llm):
    print("\n" + "=" * 60 + "\n1) DATA READER — rag/csv/sql + retry/fallback\n" + "=" * 60)
    from data_reader_agent import DataReaderAgent
    agent = DataReaderAgent(llm=llm)
    for q in ["Câte achiziții directe sunt în total?",
              "Ce email și telefon are DataPro?",
              "Ce informații avem despre contractul cu CloudNet?"]:
        res = agent.run(q)
        r = res.get("result") or {}
        print(f"\nQ: {q}\n  sursă={res.get('selected_source')} tried={res.get('sources_tried')} "
              f"→ {('OK' if r else 'EROARE: ' + str(res.get('error')))}")


def demo_orchestrator(llm):
    print("\n" + "=" * 60 + "\n2) ORCHESTRATOR + RAG — evaluate/refine\n" + "=" * 60)
    from orchestrator import Orchestrator
    app = Orchestrator(llm=llm).build_graph()
    for q in ["În ce domeniu activează CloudNet?", "Ce clauze are contractul cu DataPro?"]:
        res = app.invoke(OrchestratorState(query=q))
        print(f"\nQ: {q}\n  status={res['status']} iter={res['iteration']}\n  {res['answer'][:280]}")


def demo_analyst(llm):
    print("\n" + "=" * 60 + "\n3) ANALYST + NL2SQL — plan → query → tool → synthesize\n" + "=" * 60)
    from analyst_agent import AnalystAgent
    analyst = AnalystAgent(tables_config=_tables_config(), llm=llm)
    q = "Ia primii 15 furnizori după valoarea totală a contractelor și păstrează doar pe cei cu peste 100 de contracte."
    res = analyst.chat(q)
    print(f"\nQ: {q}\n  status={res['status']} plan={[s.id for s in res['plan']]}\n  {res['answer'][:320]}")


def demo_supervisor(llm):
    print("\n" + "=" * 60 + "\n4) SUPERVISOR — detect_intent → workeri → aggregate\n" + "=" * 60)
    from supervisor import Supervisor
    sup = Supervisor(llm=llm)
    for q in ["Care sunt primii 3 furnizori după valoarea totală?",
              "Câte achiziții directe sunt în total, și ce spune documentul despre contractul cu CloudNet?"]:
        res = sup.run(q)
        print(f"\nQ: {q}\n  intent={res.get('intent')} workers={res.get('workers')} status={res.get('status')}\n  {res.get('answer','')[:320]}")


DEMOS = {
    "datareader": demo_data_reader,
    "rag": demo_orchestrator,
    "sql": demo_analyst,
    "supervisor": demo_supervisor,
}


def main():
    llm = _make_llm()
    which = sys.argv[1:] or list(DEMOS)
    for name in which:
        fn = DEMOS.get(name)
        if not fn:
            print(f"(demo necunoscut: {name} — disponibile: {', '.join(DEMOS)})")
            continue
        try:
            fn(llm)
        except Exception as e:  # noqa: BLE001 — quota/rețea la un demo nu oprește restul
            print(f"  [demo {name} eșuat] {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
