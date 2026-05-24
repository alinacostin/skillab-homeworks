# Skillab — Homeworks

Repository containing my solutions to the Skillab AI/LLM course homeworks.

## Structure

Each homework lives in its own subfolder, scoped to its own `proiect/` (mirroring the trainer's expected layout):

```
.
├── proiect/             # Lesson-2 homework: ReAct QA agent with tools + prompts
└── README.md            # (this file)
```

When new homeworks land they'll be added as `homework2/proiect/`, `homework3/proiect/`, etc.

## Current homework: Lesson 2 — Agent QA cu Tools + Prompts

See [`proiect/README.md`](proiect/README.md) for the full description, setup, and run instructions of the lesson-2 ReAct agent.

Quick summary:
- **Tools layer** (`proiect/tools/`) — Pydantic params + `@register_tool` decorator + `ToolWrapper` (calculator, get_datetime, web_search)
- **Prompts layer** (`proiect/prompts/`) — YAML templates + Jinja2 (planner, analyst, summary, extract)
- **Agent** (`proiect/agent.py`) — `QAAgent` + `LLMFactory` + ReAct loop with parallel tool execution + 4-prompt pipeline
- **Providers** — Ollama / Gemini / Anthropic via `LLMFactory` (switch via `.env`)
