# Tema 3 — Multi-Agent System cu LangGraph

> **Stack separat de restul `proiect/`.** Acest modul folosește pachetul nativ
> **`skillab`** (SDK-uri native, `llm.generate_sync([...])`) + **LangGraph**, NU
> stack-ul LangChain din agentul ReAct al Temelor 1–2. Sunt două stack-uri care
> nu se amestecă — `multi_agent/` e izolat ca subarbore sub `proiect/`.

## Ce conține

```
src/
├── data_reader_agent.py   # un graf, surse rag/csv/sql, retry + fallback
├── orchestrator.py        # Supervizor RAG (call_rag → evaluate → answer, buclă refine)
├── rag_agent.py           # worker RAG (refine → search în pgvector)
├── nl2sql_agent.py        # worker NL2SQL (generate → validate → execute → handle_error)
├── analyst_agent.py       # Supervizor analize (make_plan → execute_step* → synthesize)
├── supervisor.py          # Supervizor central (detect_intent → workeri → aggregate)
├── data_reader, database, models, repositories, rag_service, state, json_utils
skillab-py/                # pachetul nativ `skillab` (LLM/prompts/tools) — pip install -e
prompts/                   # YAML: rag_*, nl2sql_*, analyst_*, datareader_decide
```

Supervizorul este complet **tool-driven** (fără prompt propriu): folosește tool-urile
`detect_intent` / `extract_filters` (preprocess) și `aggregate_results` / `format_response`
(postprocess) din `skillab.tools`.

## Setup

Postgres rulează pe **portul 5435** (alese diferit de celelalte stack-uri locale:
5432, 5433, 5434). Mediul Python e cel partajat al `proiect/` (`proiect/.venv`).

```bash
cd homework/proiect/multi_agent

# 1. dependențe (în venv-ul partajat al proiect/)
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m pip install -e skillab-py

# 2. Postgres + pgvector (container skillab_hw3_db, port 5435)
docker compose up -d

# 3. schema (extensie vector + tabele document_chunks / achizitii_directe / anunturi_initiere)
../.venv/bin/alembic upgrade head

# 4. date (NU sunt în git — prea mari). 
docker exec -i skillab_hw3_db pg_restore -U demo -d rag_demo --data-only --no-owner --disable-triggers \
  < /cale/către/exercise_orchestrator/data/rag_demo.dump
#   → ~694k achiziții, ~8k anunțuri, 135 chunks

# 5. cheie LLM
cp .env.example .env     
```

## Rulare

```bash
cd src
python main.py                       # toate cele 4 demo-uri (datareader, rag, sql, supervisor)
python main.py datareader            # doar Data Reader-ul 
python main.py supervisor            # doar Supervisor-ul 
```

## Note de implementare (ce am completat / reparat)

- **Tool-uri pe DataFrame** (`skillab-py/.../tools/implementations.py`): `join_data` (pd.merge), `filter_data` (== != > < >= <= contains, cu coerce numeric).
- **Tool-uri pre/postprocess** (`skillab-py/.../tools/analyst_tools.py`, deterministe, întorc JSON): `detect_intent`, `extract_filters`, `aggregate_results`, `format_response`. Apelate din `supervisor.py` (preprocess/aggregate). Analyst-ul își filtrează catalogul la
  tool-urile pe DataFrame, ca planificatorul să nu le „vadă".
- **Noduri completate**: 
  RAG `node_refine`; 
  Orchestrator `node_evaluate`/`node_answer`;
  NL2SQL `node_generate_sql`/`node_validate_sql`/`node_execute_sql`/`node_handle_error`;
  Analyst `node_make_plan`/`node_synthesize`.
- **Adăugat peste scaffold**: 
  `data_reader_agent.py`,
  `supervisor.py` (supervizorul central — alege rag/sql/csv, **wrap-uiește DataReaderAgent
  ca worker `csv`**), 
  `json_utils.py`, 
  `data/clienti.csv` (sursa CSV), 
  prompt `datareader_decide`.
- **Reparat (compatibilitate / scaffold)**:
  - `database.py` → `expire_on_commit=False` (altfel `node_search` dă
    `DetachedInstanceError` când citește chunk-urile după închiderea sesiunii).
  - `RAGAgent.run` / `NL2SQLAgent.run` reconstruiesc state-ul tipat din dict-ul întors de LangGraph (apelanții accesează `.result` / `.status`).
  - NL2SQL: `business_path` propagat pentru regulile business în prompt; plasă de siguranță `LIMIT 1000` la SELECT-uri fără limită.
  - `Supervisor` rulează workerii **secvențial** — `skillab.generate_sync` folosește un event loop partajat care nu suportă apeluri concurente.
  - `requirements.txt` += `sqlparse` (lipsea, deși `nl2sql_agent` îl importa) și `anthropic`.
  - Port 5433 → **5435** (docker-compose, alembic.ini, .env, default-urile din agenți).

## Date

`data/documents/` (docx) și `data/nl2sql_agent/*.json` sunt în git. Activele mari
**nu** sunt versionate (depășesc limita GitHub de 100MB): `data/rag_demo.dump` (120MB)
și `data/documents-sql/*.xlsx` (185MB). Baza se populează din dump-ul extern (pasul 4).
