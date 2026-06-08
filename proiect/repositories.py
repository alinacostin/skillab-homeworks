"""
Repository pattern pentru Document / DocumentChunk + similarity search pgvector.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from models import Document, DocumentChunk


class DocumentRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    # ---------- CREATE ----------
    def create_document(
        self,
        *,
        filename: str,
        doc_type: str,
        content: str,
        extracted: dict | None = None,
    ) -> Document:
        doc = Document(
            filename=filename,
            doc_type=doc_type,
            content=content,
            extracted=extracted,
        )
        self.db.add(doc)
        self.db.flush()  
        return doc

    def add_chunks(self, document_id: int, items: list[dict]) -> list[DocumentChunk]:
        chunks = [DocumentChunk(document_id=document_id, **item) for item in items]
        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def get_all(self, limit: int = 100) -> list[Document]:
        return list(self.db.execute(select(Document).limit(limit)).scalars())

    def get_by_filename(self, filename: str) -> Document | None:
        stmt = select(Document).where(Document.filename == filename)
        return self.db.execute(stmt).scalars().first()

    def get_documents(
        self, doc_type: str | None = None, limit: int = 50
    ) -> list[Document]:
        """Documente cu datele lor structurate (`extracted`), opțional filtrate pe tip."""
        stmt = select(Document)
        if doc_type:
            stmt = stmt.where(Document.doc_type == doc_type)
        return list(self.db.execute(stmt.limit(limit)).scalars())

    def count_chunks(self) -> int:
        return self.db.query(DocumentChunk).count()

    # ---------- SEARCH (pgvector) ----------
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Top-k chunk-uri după cosine similarity (= 1 - cosine_distance).
        cosine similarity ∈ [-1, 1]; pentru embeddings de text înrudit, ~[0, 1]."""
        similarity = (
            1 - DocumentChunk.embedding.cosine_distance(query_embedding)
        ).label("score")

        stmt = (
            select(DocumentChunk, similarity)
            .options(joinedload(DocumentChunk.document))  # eager load → fără N+1
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = self.db.execute(stmt).all()
        return [(chunk, float(score)) for chunk, score in rows]

    def delete_by_filename(self, filename: str) -> int:
        """Șterge documentul cu acest filename (chunk-urile cad prin ON DELETE CASCADE)."""
        result = self.db.execute(
            Document.__table__.delete().where(Document.filename == filename)
        )
        return result.rowcount

    def delete_all(self) -> int:
        """Șterge toate documentele (chunk-urile cad prin ON DELETE CASCADE)."""
        result = self.db.execute(Document.__table__.delete())
        return result.rowcount
