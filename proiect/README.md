# Document Analyst cu RAG (Tema 2)

Pipeline de **extracție + storage cu pgvector**, conectat la agentul ReAct din Tema 1.
Documentele (facturi / contracte) sunt încărcate, fragmentate, extrase
structurat, vectorizate și stocate în Postgres; agentul răspunde la întrebări despre
ele printr-un tool RAG `search_documents`.

> Construit pe agentul din Tema 1 (`main`) — vezi secțiunea *Istoric pe branch-uri*.

---

## Tema 4 — Memory, Caching & Intent Classifier

> Branch `homework4`. Trei optimizări aplicate **agentului din Tema 1** (`QAAgent`), exact
> ca în `hw8.pdf`.

| # | Parte | Ce face | Fișiere |
|---|---|---|---|
| 1 | **Conversation Memory** | `ConversationMemory` persistentă în PostgreSQL: `ask()` face load → invoke → save pe `session_id`; contextul supraviețuiește restart-urilor | `models.py` (`ChatSession`/`ChatMessage`), `memory.py` (`PersistentMemory`), `alembic/versions/0003_*`, `agent.py` |
| 2 | **Prompt Caching** | Anthropic `cache_control: ephemeral` pe system prompt (prefix static) → ~90% mai puțini tokeni de input la apelurile următoare; usage măsurat | `agent.py` (`_system_message`), `benchmark_prompt_cache.py` |
| 3 | **Intent Classifier** | TF-IDF + LogisticRegression (`search`/`extract`/`summarize`) înlocuiește un apel LLM de routing; fallback la LLM când `confidence < prag` | `intent/` (`intent_data.py`, `train_intent.py`, `classifier.py`), `benchmark_intent.py`, `agent.py` |

### Cum funcționează în `agent.py` (`ask`)

```
0. intent  → detect_intent (sklearn)   ; sub prag → fallback LLM
1. memory  → load_messages(session_id) ; istoricul injectat în loop-ul ReAct
2. ReAct   → system prompt cu cache_control (Anthropic) ; usage acumulat
3. analyst → summary → extract (extract rulează doar pentru intent „extract")
4. memory  → save_turn(session_id, user, assistant)   ; persistat în Postgres
```

### Rulare (Tema 4)

```bash
docker compose up -d && alembic upgrade head      # creează și tabelele memory (migrația 0003)

# Part 3 — antrenează classifier-ul (o singură dată) + benchmark LLM vs sklearn
python -m intent.train_intent
python benchmark_intent.py

# Part 2 — măsoară prompt caching (cere ANTHROPIC_API_KEY)
DEFAULT_PROVIDER=anthropic DEFAULT_MODEL= python benchmark_prompt_cache.py

# Part 1 — memory care supraviețuiește „restart-ului"
DEFAULT_PROVIDER=anthropic DEFAULT_MODEL= python demo.py memory
#   sau manual, două procese separate:
DEFAULT_PROVIDER=anthropic DEFAULT_MODEL= SESSION_ID=andrei python agent.py "Mă numesc Andrei, lucrez cu DataPro."
DEFAULT_PROVIDER=anthropic DEFAULT_MODEL= SESSION_ID=andrei python agent.py "Cum mă numesc?"
```

### Rezultate măsurate (validare locală)

- **Intent**: held-out accuracy 100% (24 exemple); sklearn ~0.4 ms/query vs LLM ~sute de ms; cost ~$0.01 vs ~$20 / 1000 calls.
- **Prompt caching**: prefix de 7212 tokeni scris în cache (apel 1) și servit din cache (apel 2) → **~90% reducere** pe inputul cache-uit. Pe agentul real, system prompt-ul (planner + tool catalog ≈ 2157 tokeni, peste pragul de 2048 al Sonnet 4.x) se cache-uiește efectiv.
- **Memory**: turele se persistă în `chat_messages`; un agent nou (proces nou) reîncarcă istoricul din Postgres.

### Note / decizii

- **Integrare în loop-ul ReAct, nu într-un nod LangGraph.** Brief-ul zice „nod LangGraph", dar agentul Temei 1 e un loop ReAct hand-built (convențiile cursului interzic LangGraph în acest stack). Memoria intră în `ask()`/`_react_loop`. Varianta LangGraph-native (`MessagesState` + checkpointer) există deja în `multi_agent/` din Tema 3.
- **Caching = doar Anthropic.** Auto-pornit când `DEFAULT_PROVIDER=anthropic`. Prag minim de prefix: 1024 tok (Claude 3.x) / 2048 (Sonnet 4.x) — sub prag nu se activează (fără eroare). ⚠️ `DEFAULT_MODEL` se trimite verbatim provider-ului → lasă-l gol când treci pe anthropic (altfel un `llama3.2` rămas în `.env` dă 404).
- **Etichete intent**: `search/extract/summarize`, conform brief-ului.

---

## Flow

```
pipeline.py:  load(file) ─► split(800,100) ─► extract(Invoice|Contract) ─► embed ─► store
              (registry      RecursiveChar      LLM.with_structured_       sentence-   Document
               pe extensie)  TextSplitter       output)                   transformers + Chunks)

agent.py:     întrebare ─► ReAct loop ─► search_documents / query_documents ─► pgvector ─► răspuns citat
```

## Cele 4 părți ale temei

| # | Parte | Fișiere |
|---|---|---|
| 1 | **Extraction pipeline** — loader registry PDF/DOCX/TXT/CSV, chunking, schemă Pydantic, salvare JSON | `loaders.py`, `chunking.py`, `schemas.py`, `pipeline.py`, `prompts/doc_extract.yaml` |
| 2 | **Postgres + Repository** — pgvector via Docker + Alembic, modele `Document` 1→N `DocumentChunk` | `docker-compose.yml`, `alembic/`, `database.py`, `models.py`, `repositories.py` |
| 3 | **RAG cu embeddings** — sentence-transformers per chunk, cosine + HNSW, `RAGService.search` | `rag_service.py`, `create_index.py` (HNSW), `repositories.py` |
| 4 | **Conectare la agent** — două tool-uri pe documente (`search_documents` semantic + `query_documents` pe datele extrase) adăugate la agentul existent | `tools/rag_tools.py`, `tools/params_models.py`, `prompts/planner.yaml` |

## Structură

```
proiect/
├── agent.py              # QAAgent + LLMFactory + ReAct loop + intent/memory/cache (T1 + T4)
├── pipeline.py           # load → chunk → extract → store  (entry point)
├── demo.py               # demo end-to-end: `rag` (default) + `memory` (T4)
│
├── memory.py             # T4·1: PersistentMemory (conversation memory → Postgres)
├── intent/               # T4·3: TF-IDF + LogReg (intent_data, train_intent, classifier)
├── benchmark_intent.py   # T4·3: LLM vs sklearn (latență/cost/accuracy)
├── benchmark_prompt_cache.py  # T4·2: măsoară prompt caching Anthropic
│
├── loaders.py            # LOADER_REGISTRY pe extensie (txt/pdf/docx/csv)
├── chunking.py           # split(docs, 800, 100)
├── schemas.py            # Invoice + Contract (Pydantic)
│
├── database.py           # engine + transaction() context manager
├── models.py             # Document (1) → DocumentChunk (N), Vector(384)
├── repositories.py       # DocumentRepository + similarity_search (cosine)
├── rag_service.py        # RAGService: embed + search(query, top_k)
│
├── tools/                # din Tema 1 + rag_tools.py (search_documents + query_documents)
├── prompts/              # din Tema 1 + doc_extract.yaml
│
├── alembic/ + alembic.ini   # migrația: extensia vector + tabele (documents, chunks)
├── create_index.py          # indexul HNSW, separat de migrație
├── docker-compose.yml       # pgvector pe portul 5434
├── data/documents/          # corpus demo (facturi + contracte)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # pune GOOGLE_API_KEY (Gemini) pentru extracție + răspuns agent

# 1. Pornește Postgres + pgvector (port 5434)
docker compose up -d

# 2. Creează schema (extensie vector + tabele)
alembic upgrade head

# 3. Construiește indexul HNSW (separat: CREATE INDEX CONCURRENTLY nu merge în tranzacția Alembic)
python create_index.py
```

## Rulare

```bash
# Ingest corpusul demo (load → chunk → extract → store)
python pipeline.py
#   sau fișiere specifice:  python pipeline.py data/documents/factura_001.txt

# Întrebări către agent (folosește tool-ul search_documents)
python agent.py "Ce clauze de reziliere/confidențialitate avem?"
python agent.py "Care e termenul de plată al facturilor?"

# Demo end-to-end (ingest + 3 întrebări)
python demo.py
```

> **Fără cheie LLM?** Storage-ul și căutarea RAG merg și fără: extracția structurată
> e *best-effort* (dacă LLM-ul lipsește, `extracted` rămâne `null`, dar documentul și
> chunk-urile se stochează normal). Doar extracția câmpurilor și răspunsul final al
> agentului au nevoie de o cheie.

## Convenții respectate (pe lângă cele din Tema 1)

- **Loader = Registry Pattern pe extensie** (`LOADER_REGISTRY`: txt/pdf/docx/csv), importuri lazy
  per loader. `CSVLoader` emite un Document per rând (potrivit pentru tabele, nu pentru proză lungă);
  fără backend extra (modulul `csv` din stdlib).
- **Tool-uri pe documente prin `@register_tool`** (nu `@tool`-ul LangChain) — ajung la agent prin
  `ToolWrapper.catalog()` → `bind_tools()`, exact ca tool-urile existente. Două complementare:
  `search_documents` (semantic, text liber) și `query_documents` (datele structurate extrase,
  ca extracția să fie efectiv folosită la răspuns, nu doar salvată).
- **Ingest idempotent**: re-procesarea aceluiași fișier înlocuiește documentul (constrângere
  `UNIQUE(filename)` + `delete_by_filename`), nu îl duplică (migrația `0002`).
- **Repository Pattern** ascunde SQL-ul; commit doar în `transaction()`, `flush()` în repo.
- **`Document` 1→N `DocumentChunk`** cu `ondelete=CASCADE` + `UniqueConstraint(document_id, chunk_index)`.
- **pgvector**: `Vector(384)`, cosine (`1 - cosine_distance`), **HNSW** `vector_cosine_ops`.
- **Embeddings multilingve** (`paraphrase-multilingual-MiniLM-L12-v2`) pentru documente RO,
  încărcate lazy o singură dată (singleton per proces).
- **Alembic** pentru schema, cu `CREATE EXTENSION` manual în `upgrade()`; indexul **HNSW**
  se construiește separat în `create_index.py` via `engine.begin()`, fiindcă
  `CREATE INDEX CONCURRENTLY` nu poate rula în tranzacția unei migrații.
- **Prompt de extracție în YAML** (`doc_extract.yaml`), nu hardcodat — Registry Pattern pentru prompt-uri.

## Istoric pe branch-uri

Fiecare temă e un branch peste cea anterioară (folderul `proiect/` evoluează):

- `main` — Tema 1: agent ReAct cu tools + prompts.
- `homework2` — Tema 2: Document Analyst cu RAG, peste agentul din `main`.
- `homework3` — Tema 3: sistem multi-agent cu LangGraph (subarborele `multi_agent/`).
- `homework4` — Tema 4: **memory + caching + intent classifier**, aplicate agentului din Tema 1 (vezi secțiunea *Tema 4* de mai sus).
