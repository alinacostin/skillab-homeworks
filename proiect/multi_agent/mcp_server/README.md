# Tema 5 — MCP & Guardrails

Împachetează agenții Data Analyst și Supervizor ca **tool-uri MCP** și protejează
serverul cu **guardrails** (input validation + prompt injection). Construit peste
subarborele `multi_agent/` (stack-ul nativ `skillab` + LangGraph).

## Ce expune

Un singur server FastMCP (`document-agents`, transport **stdio**) cu **două tool-uri**:

| Tool | Agent | Apel | Pentru |
|---|---|---|---|
| `data_analyst` | `AnalystAgent` (`src/analyst_agent.py`) | `.chat(question)` | întrebări analitice pe date structurate (top furnizori, agregări, filtrări) |
| `orchestrator` | `Supervisor` (`src/supervisor.py`) | `.run(query)` | întrebări generale; supervizorul alege singur workerii (rag/sql/csv) și agregă |

Fiecare tool întoarce output **JSON-safe** (proiecție din state-urile agenților, care
conțin DataFrame-uri / obiecte Pydantic).

## Fluxul unui apel (ordinea contează)

1. **Input validation** (`schemas.py`) — Pydantic verifică **tip** (`str`), **dimensiune**
   (`max_length=4000`) și **câmpuri permise** (`extra="forbid"`). Eșec → `McpError -32602`.
2. **Guardrail prompt-injection** (`guardrails.py`) — `InputValidator` în 3 straturi:
   - **regex** (mereu pornit) — tipare ro+en;
   - **embedding similarity** (opțional, `GUARDRAIL_EMBEDDING=1`) — cosine > 0.75 vs.
     injecții cunoscute, cu `all-MiniLM-L6-v2`;
   - **LLM-as-Judge** (opțional, `GUARDRAIL_LLM_JUDGE=1`) — prompt în
     `prompts/guardrail_judge.yaml`.

   Input periculos → **refuz structurat** (`status="blocked"`, `reason=...`), agentul NU e apelat.
3. **Apel agent** — eșec de infrastructură (DB/LLM) → `McpError -32000`.

## Fișiere

```
mcp_server/
├── server.py        # FastMCP + cele 2 tool-uri; rulează stdio (sau http)
├── schemas.py       # modele Pydantic in/out (input validation)
├── guardrails.py    # InputValidator (regex + embedding + LLM-judge) + ValidationResult
├── runners.py       # singletoni lazy AnalystAgent/Supervisor + proiecție JSON-safe
├── client_demo.py   # client stdio: list_tools + 5 apeluri (injection, validare, valid)
├── test_guardrails.py  # test OFFLINE (fără DB/LLM) pentru guardrails + validare
├── claude_mcp_config.example.json   # config gata de copiat pentru Claude Code
└── _bootstrap.py    # pune skillab-py/src și src/ pe sys.path
prompts/guardrail_judge.yaml         # promptul LLM-judge (în registry-ul existent)
```

## Rulare

Presupune mediul `multi_agent/` deja pus la punct (vezi `../README.md`): venv-ul
`proiect/.venv`, `pip install -r ../requirements.txt` (acum include `fastmcp`),
`docker compose up -d` (Postgres pe **5435**), `alembic upgrade head`, datele restaurate
și o cheie LLM în `../.env`.

```bash
cd homework/proiect/multi_agent/mcp_server

# 1. test guardrails — OFFLINE, nu cere DB/LLM (cod 0 = trece)
../../.venv/bin/python test_guardrails.py

# 2. demo client ↔ server prin stdio (pornește singur server.py ca subproces)
../../.venv/bin/python client_demo.py

# 3. server brut (pentru Claude Code / Claude Desktop)
../../.venv/bin/python server.py          # stdio (default)
../../.venv/bin/python server.py http      # http://127.0.0.1:8000 (debug)
```

### Din Claude Code

Copiază `claude_mcp_config.example.json` în `.mcp.json` (la rădăcina proiectului tău
Claude Code) — sau adaugă serverul cu:

```bash
claude mcp add document-agents -- \
  /Users/alinacostin/learning/Skillab/homework/proiect/.venv/bin/python \
  /Users/alinacostin/learning/Skillab/homework/proiect/multi_agent/mcp_server/server.py
```

Apoi în sesiune apar tool-urile `document-agents:data_analyst` și `document-agents:orchestrator`.

## Note

- Stack izolat: native `skillab` + LangGraph (ca tot `multi_agent/`), **nu** stack-ul
  LangChain din Temele 1–2.
- Transportul stdio folosește **stdout** pentru protocol → serverul logează doar pe
  **stderr** și nu folosește `print`.
- Serverul pornește și listează tool-urile **fără** DB/LLM; agenții se construiesc lazy
  la primul apel curat.
