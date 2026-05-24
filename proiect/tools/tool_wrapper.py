"""ToolWrapper.

call(name, args): lookup → Pydantic validate → execute → return string.
catalog(): listă de StructuredTool pentru llm.bind_tools(...).
catalog_anthropic(): format Anthropic JSON Schema nativ.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from .registry import TOOL_REGISTRY


class ToolWrapper:
    @staticmethod
    def call(name: str, args: dict) -> str:
        if name not in TOOL_REGISTRY:
            return f"Eroare: tool '{name}' nu există."

        tool = TOOL_REGISTRY[name]

        try:
            params = tool["params_model"](**args)
        except Exception as e:
            return f"Eroare validare pentru '{name}': {e}"

        try:
            return str(tool["func"](params))
        except Exception as e:
            return f"Eroare execuție '{name}': {e}"

    @staticmethod
    def catalog() -> list[StructuredTool]:
        """LangChain-compatible catalog — direct bindable cu llm.bind_tools(catalog())."""
        tools = []
        for name, entry in TOOL_REGISTRY.items():
            params_model = entry["params_model"]
            func = entry["func"]

            def _make_runner(_func, _model):
                def _run(**kwargs):
                    return str(_func(_model(**kwargs)))
                return _run

            tools.append(
                StructuredTool.from_function(
                    func=_make_runner(func, params_model),
                    name=name,
                    description=entry["description"],
                    args_schema=params_model,
                )
            )
        return tools

    @staticmethod
    def catalog_anthropic() -> list[dict[str, Any]]:
        """Format Anthropic native — JSON Schema per tool."""
        return [
            {
                "name": name,
                "description": entry["description"],
                "input_schema": entry["params_model"].model_json_schema(),
            }
            for name, entry in TOOL_REGISTRY.items()
        ]
