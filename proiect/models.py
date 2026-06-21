"""
SQLAlchemy models: Document (1) → DocumentChunk (N), cu pgvector.
"""

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from database import Base

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2


class Document(Base):
    """Document procesat — text integral + rezultatul extracției structurate."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)  
    doc_type = Column(String(32), nullable=False)  
    content = Column(Text, nullable=False)  
    extracted = Column(JSONB, nullable=True)  
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )

    __table_args__ = (
        UniqueConstraint("filename", name="uq_documents_filename"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} {self.doc_type} '{self.filename}'>"


class DocumentChunk(Base):
    """Fragment de document cu embedding pentru similarity search."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_idx"),
        Index("ix_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"


# ============================================================
# Conversation Memory — sessions (1) → chat_messages (N)
# Persistă istoricul conversațiilor pentru long-term storage / context
# între request-uri (supraviețuiește restart-urilor).
# ============================================================


class ChatSession(Base):
    """O conversație unică (un session_id). Ține user-ul și metadata."""

    __tablename__ = "sessions"

    # session_id ca string (UUID sau nume liber) — ales de apelant.
    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 'metadata' e rezervat în SQLAlchemy → folosim 'meta'.
    meta = Column(JSONB, nullable=True)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id!r} user={self.user_id!r}>"


class ChatMessage(Base):
    """Un mesaj din conversație: role (user/assistant) + content + timestamp."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage {self.role} sess={self.session_id!r}>"
