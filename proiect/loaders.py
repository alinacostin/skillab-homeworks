"""
Loader cu registry pe extensie (PDF / DOCX / TXT / CSV) — L3.
"""

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document as LCDocument


def _load_txt(path: Path) -> list[LCDocument]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8").load()


def _load_pdf(path: Path) -> list[LCDocument]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _load_docx(path: Path) -> list[LCDocument]:
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path)).load()


def _load_csv(path: Path) -> list[LCDocument]:
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(str(path)).load()


LOADER_REGISTRY: dict[str, Callable[[Path], list[LCDocument]]] = {
    "txt": _load_txt,
    "pdf": _load_pdf,
    "docx": _load_docx,
    "csv": _load_csv,
}


def load(path: str | Path) -> list[LCDocument]:
    path = Path(path)
    ext = path.suffix.lstrip(".").lower()
    if ext not in LOADER_REGISTRY:
        disponibile = ", ".join(sorted(LOADER_REGISTRY))
        raise ValueError(f"Extensie nesuportată: '.{ext}'. Disponibile: {disponibile}")
    return LOADER_REGISTRY[ext](path)
