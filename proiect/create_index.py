"""
create_index.py — construiește indexul HNSW pentru cosine similarity (L4, slide 70).
"""

import logging

from dotenv import load_dotenv

load_dotenv()  # înainte de importul `engine` — ca să prindă DATABASE_URL din .env

from sqlalchemy import text  # noqa: E402

from database import engine  # noqa: E402

logger = logging.getLogger(__name__)

INDEX_NAME = "ix_chunks_embedding_hnsw"
TABLE = "document_chunks"
COLUMN = "embedding"


def create_index() -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))

        conn.execute(
            text(
                f"""
                CREATE INDEX {INDEX_NAME}
                ON {TABLE}
                USING hnsw ({COLUMN} vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        )

        size = conn.execute(
            text("SELECT pg_size_pretty(pg_relation_size(:idx))"),
            {"idx": INDEX_NAME},
        ).scalar()
        logger.info("Index %s creat, size=%s", INDEX_NAME, size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_index()
