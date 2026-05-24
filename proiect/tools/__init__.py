"""Tools package. Importul side-effect din basic_tools populează TOOL_REGISTRY."""

from . import basic_tools  # noqa: F401  — side-effect: înregistrează tool-urile
from .registry import TOOL_REGISTRY, register_tool
from .tool_wrapper import ToolWrapper

__all__ = ["ToolWrapper", "register_tool", "TOOL_REGISTRY"]
