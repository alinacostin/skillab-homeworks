# Agent QA cu Tools + Prompts

Tema 1 (lecția 2): agent ReAct care răspunde la întrebări factuale folosind
tool-uri (`calculator`, `get_datetime`, `web_search`) și un pipeline de
prompt-uri YAML (planner → analyst → summary → extract).

## Structură

```
proiect/
├── agent.py                   # QAAgent + LLMFactory + main()
├── tools/
│   ├── __init__.py
│   ├── registry.py            # TOOL_REGISTRY + @register_tool
│   ├── params_models.py       # Pydantic BaseModel per tool
│   ├── basic_tools.py         # calculator, get_datetime, web_search
│   └── tool_wrapper.py        # ToolWrapper.call() + .catalog()
├── prompts/
│   ├── __init__.py
│   ├── registry.py            # PromptRegistry (YAML + Jinja2)
│   ├── planner.yaml
│   ├── analyst.yaml
│   ├── summary.yaml
│   └── extract.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# editează .env și pune cheia ta (GOOGLE_API_KEY pentru Gemini)
```

## Rulare

```bash
python agent.py "Câte zile sunt între azi și 1 ianuarie 2027? Caută pe web ce sărbătoare e atunci."
```

Sau interactiv:

```bash
python agent.py
# Întrebare: <scrie întrebarea>
```

## Switch provider

În `.env`:

```
DEFAULT_PROVIDER=ollama       # local, fără cheie (necesită ollama serve + llama3.2)
DEFAULT_PROVIDER=gemini       # implicit, free tier
DEFAULT_PROVIDER=anthropic    # paid
```

> **Notă:** doar unele modele Ollama suportă tool-calling de încredere
> (`llama3.2`, `qwen2.5`). Gemini e cea mai sigură alegere pentru demo.

## Cum funcționează pipeline-ul

```
Întrebare utilizator
   │
   ▼
[planner system + tools] → ReAct loop (Think/Act/Observe ×N, tool calls în paralel)
   │
   ▼ raw_answer + trace
[analyst]  → critică & rafinează
   │
   ▼ refined_answer
[summary]  → format final user-facing
   │
   ▼ summary text
[extract]  → {answer, key_facts, sources, tools_used} (JSON)
```

Output-ul `agent.ask(question)` conține:

- `answer` — textul formatat de `summary`
- `structured` — JSON-ul de la `extract`
- `trace` — lista apelurilor de tool-uri (iterație, nume, args, rezultat)
- `raw_answer`, `refined_answer` — pași intermediari, utili la debugging

## Convenții respectate

- `LLMFactory.create(provider, **kwargs)` — Factory pattern pentru switch între provideri
- Pydantic `BaseModel` per tool cu `Field(description=...)` — schema vizibilă pentru LLM
- `@register_tool` validează BaseModel unic + docstring ≥ 15 caractere
- `ToolWrapper.call()` (lookup → validate → execute → return str) + `.catalog()`
- Anatomia YAML: `name`, `version`, `description`, `prompt`
- `PromptTemplate` dataclass `frozen=True`
- `PromptRegistry._load()` + `.render()` cu Jinja2
- `get_prompt_registry()` singleton
- System prompt cu 5 secțiuni (rol / obiectiv / context / constrângeri / format)
- Reminder block la final (Lost in the Middle)
- `react_loop()` Think → Act → Observe
- `asyncio.gather` pentru tool calls multiple în paralel
- Erori human-readable, nu stack traces; loop continuă pe eroare
