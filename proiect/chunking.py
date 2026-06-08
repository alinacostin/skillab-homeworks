"""
Chunking cu RecursiveCharacterTextSplitter 
"""

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split(
    docs: list[LCDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[LCDocument]:
    """Împarte documentele în chunk-uri suprapuse."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
