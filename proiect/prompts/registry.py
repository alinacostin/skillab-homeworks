"""PromptTemplate + PromptRegistry + singleton."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    prompt: str
    description: str = ""


class PromptRegistry:
    def __init__(self, folder: str):
        self._folder = Path(folder)
        self._templates: dict[str, PromptTemplate] = self._load()

    def _load(self) -> dict[str, PromptTemplate]:
        templates: dict[str, PromptTemplate] = {}
        for path in self._folder.rglob("*.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            tpl = PromptTemplate(
                name=data["name"],
                version=str(data["version"]),
                prompt=data["prompt"],
                description=data.get("description", ""),
            )
            templates[tpl.name] = tpl
        return templates

    def render(self, name: str, **variabile) -> str:
        if name not in self._templates:
            raise KeyError(f"Prompt '{name}' nu există în registry.")
        template = self._templates[name]
        return Template(template.prompt).render(**variabile)

    def list_templates(self) -> list[str]:
        return sorted(self._templates.keys())

    def reload(self) -> None:
        self._templates = self._load()


_instance: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Singleton — o singură încărcare YAML, reutilizată în tot codul."""
    global _instance
    if _instance is None:
        folder = Path(__file__).parent
        _instance = PromptRegistry(folder=str(folder))
    return _instance
