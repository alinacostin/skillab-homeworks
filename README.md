# Skillab — Homeworks

Repository containing my solutions to the Skillab AI/LLM course homeworks.

## Model pe branch-uri

Fiecare temă este rezolvată pe **propriul branch**, pornit din branch-ul temei anterioare.
Folderul `proiect/` evoluează cumulativ — fiecare temă adaugă peste agentul/infra-ul de dinainte
(tema 2 cheamă agentul din tema 1, tema 3 va construi peste tema 2 etc.).

| Temă | Lecție | Branch | Pornit din | Subiect |
|---|---|---|---|---|
| 1 | L2 (`hw2.pdf`) | `main` | — | Agent QA ReAct cu tools + prompts |
| 2 | L3+L4 (`hw4.pdf`) | `homework2` | `main` | Document Analyst cu RAG (extracție + pgvector + tool RAG) |
| 3 | … | `homework3` | `homework2` | (urmează) |

```
.
├── proiect/             # soluția temei de pe branch-ul curent
└── README.md            # (acest fișier)
```

## Tema curentă pe acest branch

Vezi [`proiect/README.md`](proiect/README.md) pentru descrierea completă, setup și rulare.

- **`main`** — Agent QA ReAct: tools (`@register_tool` + `ToolWrapper`), prompts YAML
  (planner → analyst → summary → extract), `LLMFactory` (Ollama / Gemini / Anthropic).
- **`homework2`** — Document Analyst cu RAG peste agentul din `main`:
  - Extraction pipeline (L3): loader registry PDF/DOCX/TXT → chunking → `Invoice`/`Contract`
    cu `with_structured_output` → JSON.
  - Storage (L4): Postgres + pgvector via Docker + Alembic; `Document` 1→N `DocumentChunk`;
    Repository Pattern.
  - RAG (L4): embeddings sentence-transformers per chunk, cosine + HNSW, `RAGService.search`.
  - Integrare: RAG wrappat ca tool `search_documents` și adăugat la agentul existent.
