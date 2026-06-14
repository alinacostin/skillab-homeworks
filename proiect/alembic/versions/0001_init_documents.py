"""init: documents + document_chunks (pgvector; HNSW separat în create_index.py)

Revision ID: 0001
Revises:
Create Date: 2026-06-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2


def upgrade() -> None:
    # Extensia pgvector — autogenerate NU o detectează, o adăugăm manual.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("extracted", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_filename", "documents", ["filename"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_idx"),
    )
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])
    # Indexul HNSW NU se creează aici — vezi create_index.py (slide 70):
    # CREATE INDEX CONCURRENTLY nu poate rula în tranzacția unei migrații Alembic.


def downgrade() -> None:
    # Indexul HNSW (creat de create_index.py) e șters automat odată cu tabela.
    op.drop_index("ix_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_filename", table_name="documents")
    op.drop_table("documents")
