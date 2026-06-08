# Document Analyst cu RAG (Tema 2 — L3 + L4)

Pipeline de **extracție + storage cu pgvector**, conectat la agentul ReAct din Tema 1
(L1–L2). Documentele (facturi / contracte) sunt încărcate, fragmentate, extrase
structurat, vectorizate și stocate în Postgres; agentul răspunde la întrebări despre
ele printr-un tool RAG `search_documents`.

> Construit pe agentul din Tema 1 (`main`) — vezi secțiunea *Istoric pe branch-uri*.

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
| 1 | **Extraction pipeline (L3)** — loader registry PDF/DOCX/TXT/CSV, chunking, schemă Pydantic, salvare JSON | `loaders.py`, `chunking.py`, `schemas.py`, `pipeline.py`, `prompts/doc_extract.yaml` |
| 2 | **Postgres + Repository (L4)** — pgvector via Docker + Alembic, modele `Document` 1→N `DocumentChunk` | `docker-compose.yml`, `alembic/`, `database.py`, `models.py`, `repositories.py` |
| 3 | **RAG cu embeddings (L4)** — sentence-transformers per chunk, cosine + HNSW, `RAGService.search` | `rag_service.py`, `create_index.py` (HNSW), `repositories.py` |
| 4 | **Conectare la agent (L1–L2)** — două tool-uri pe documente (`search_documents` semantic + `query_documents` pe datele extrase) adăugate la agentul existent | `tools/rag_tools.py`, `tools/params_models.py`, `prompts/planner.yaml` |

## Structură

```
proiect/
├── agent.py              # QAAgent + LLMFactory + ReAct loop (din Tema 1, neschimbat)
├── pipeline.py           # load → chunk → extract → store  (entry point L3+L4)
├── demo.py               # ingest + 3 întrebări către agent (demo end-to-end)
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
├── create_index.py          # indexul HNSW, separat de migrație (slide 70)
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
  ca extracția din L3 să fie efectiv folosită la răspuns, nu doar salvată).
- **Ingest idempotent**: re-procesarea aceluiași fișier înlocuiește documentul (constrângere
  `UNIQUE(filename)` + `delete_by_filename`), nu îl duplică (migrația `0002`).
- **Repository Pattern** ascunde SQL-ul; commit doar în `transaction()`, `flush()` în repo.
- **`Document` 1→N `DocumentChunk`** cu `ondelete=CASCADE` + `UniqueConstraint(document_id, chunk_index)`.
- **pgvector**: `Vector(384)`, cosine (`1 - cosine_distance`), **HNSW** `vector_cosine_ops`.
- **Embeddings multilingve** (`paraphrase-multilingual-MiniLM-L12-v2`) pentru documente RO,
  încărcate lazy o singură dată (singleton per proces).
- **Alembic** pentru schema (slide 55), cu `CREATE EXTENSION` manual în `upgrade()`; indexul **HNSW**
  se construiește separat în `create_index.py` via `engine.begin()` (slide 70), fiindcă
  `CREATE INDEX CONCURRENTLY` nu poate rula în tranzacția unei migrații.
- **Prompt de extracție în YAML** (`doc_extract.yaml`), nu hardcodat — Registry Pattern pentru prompt-uri.

## Istoric pe branch-uri

Fiecare temă e un branch peste cea anterioară (folderul `proiect/` evoluează):

- `main` — Tema 1 (L2): agent ReAct cu tools + prompts.
- `homework2` — Tema 2 (L3+L4): **acest** Document Analyst cu RAG, peste agentul din `main`.
- `homework3` — va porni din `homework2`, ș.a.m.d.
