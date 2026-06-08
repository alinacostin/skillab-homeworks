"""
RAGService: embeddings (sentence-transformers) + similarity search
"""

from sqlalchemy.orm import Session

from models import DocumentChunk
from repositories import DocumentRepository

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  

class RAGService:
    """Embedding + retrieval peste documentele stocate în pgvector."""

    _model = None  

    def __init__(self, db: Session | None = None) -> None:
        self.repo = DocumentRepository(db) if db is not None else None

    @property
    def model(self):
        """Lazy loading — modelul se descarcă/încarcă o singură dată."""
        if RAGService._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Încărcare model embeddings {MODEL_NAME}...")
            RAGService._model = SentenceTransformer(MODEL_NAME)
        return RAGService._model

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Caută cele mai relevante chunk-uri pentru un query."""
        return self.repo.similarity_search(self.embed(query), top_k=top_k)

    def get_context(
        self, query: str, top_k: int = 3, threshold: float = 0.2
    ) -> str:
        """Context formatat pentru LLM: fragmente relevante cu sursă + scor."""
        results = self.search(query, top_k=top_k)
        relevant = [(c, s) for c, s in results if s >= threshold]
        if not relevant:
            return ""

        linii = []
        for chunk, score in relevant:
            sursa = chunk.document.filename if chunk.document else f"doc#{chunk.document_id}"
            linii.append(f"[{sursa} · score={score:.2f}] {chunk.content}")
        return "\n\n".join(linii)
