"""
pipeline.py — load → chunk → extract → store 
"""

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from chunking import split
from database import transaction
from loaders import load
from models import Document
from prompts import get_prompt_registry
from rag_service import RAGService
from repositories import DocumentRepository
from schemas import SCHEMA_BY_TYPE

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EXTRACTED_DIR = DATA_DIR / "extracted"


def detect_doc_type(path: Path, text: str) -> str:
    head = text[:200].lower()
    name = path.stem.lower()
    if name.startswith("contract") or "contract" in head:
        return "contract"
    if name.startswith("factura") or "factura" in head:
        return "factura"
    print(
        f"  [detect] tip nedetectat pentru {path.name} — presupun 'factura'",
        file=sys.stderr,
    )
    return "factura"  # fallback


def extract_structured(text: str, doc_type: str) -> dict[str, Any] | None:
    schema = SCHEMA_BY_TYPE[doc_type]
    try:
        from agent import LLMFactory  

        llm = LLMFactory.create(temperature=0)
        structured = llm.with_structured_output(schema)
        prompt = get_prompt_registry().render(
            "doc_extract", doc_type=doc_type, document_text=text
        )
        result = structured.invoke(prompt)
        return result.model_dump()
    except Exception as e:
        print(f"  [extract] sărit (LLM indisponibil): {e}", file=sys.stderr)
        return None


def save_json(doc_type: str, numar: str, data: dict[str, Any]) -> Path:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    safe = (numar or "necunoscut").replace("/", "-")
    out = EXTRACTED_DIR / f"{doc_type}_{safe}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def process(file: Path) -> Document:
    """Procesează un fișier: load → chunk → extract → store. Returnează Document."""
    file = Path(file)
    print(f"\n→ {file.name}")

    docs = load(file)
    full_text = "\n".join(d.page_content for d in docs)

    chunks = split(docs, 800, 100)
    print(f"  {len(chunks)} chunk-uri")

    doc_type = detect_doc_type(file, full_text)
    extracted = extract_structured(full_text, doc_type)
    if extracted:
        numar = extracted.get("numar", "")
        out = save_json(doc_type, numar, extracted)
        print(f"  extras [{doc_type}] {numar} → {out.name}")


    embeddings = RAGService().embed_batch([c.page_content for c in chunks])

    with transaction() as db:
        repo = DocumentRepository(db)
        if repo.get_by_filename(file.name):
            repo.delete_by_filename(file.name)
            print("  (exista deja — înlocuit)")
        document = repo.create_document(
            filename=file.name,
            doc_type=doc_type,
            content=full_text,
            extracted=extracted,
        )
        repo.add_chunks(
            document.id,
            [
                {"chunk_index": i, "content": c.page_content, "embedding": emb}
                for i, (c, emb) in enumerate(zip(chunks, embeddings))
            ],
        )
        print(f"  stocat Document id={document.id} cu {len(chunks)} chunk-uri")
        return document


def ingest(paths: list[Path]) -> list[Document]:
    return [process(p) for p in paths]


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted(DOCUMENTS_DIR.glob("*.txt"))
        if not paths:
            print(
                f"Niciun fișier în {DOCUMENTS_DIR}.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Ingest {len(paths)} fișiere...")
    ingest(paths)
    print("\n✓ Gata. Pune întrebări cu: python agent.py \"...\"  sau  python demo.py")


if __name__ == "__main__":
    main()
