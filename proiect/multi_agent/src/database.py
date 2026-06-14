"""
Database - Connection + Transaction Context Manager
"""
import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://demo:demo123@localhost:5435/rag_demo"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
# expire_on_commit=False: obiectele ORM rămân utilizabile după închiderea sesiunii.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def transaction() -> Generator[Session, None, None]:
    """
    Context manager pentru tranzacții.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Returnează o sesiune nouă (trebuie închisă manual)."""
    return SessionLocal()
