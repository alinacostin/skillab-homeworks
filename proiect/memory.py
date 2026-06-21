"""Conversation Memory persistentă în PostgreSQL.

`PersistentMemory` = manager peste un `ChatMessageRepository` (Repository pattern),
folosind context manager-ul `transaction()` din `database.py` ca unit of work. Pattern-ul
e load → invoke → save, cu o fereastră glisantă (ultimele N mesaje). Memoria
supraviețuiește restart-urilor fiindcă trăiește în DB, nu în RAM.

    mem = PersistentMemory(window=10)
    history = mem.load_messages("andrei")          # [{role, content}, ...] cronologic
    ...                                            # invoke LLM cu history injectat
    mem.save_turn("andrei", user_msg, assistant_msg)
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from database import transaction
from models import ChatMessage, ChatSession


class ChatMessageRepository:
    """Toată logica de DB pentru sesiuni/mesaje, izolată aici (Repository pattern)."""

    def __init__(self, db: DBSession):
        self.db = db

    def ensure_session(self, session_id: str, user_id: str = "default") -> None:
        """Creează sesiunea dacă nu există (FK-ul cere ca ea să existe înainte de mesaje)."""
        if self.db.get(ChatSession, session_id) is None:
            self.db.add(ChatSession(id=session_id, user_id=user_id))

    def add(self, session_id: str, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(session_id=session_id, role=role, content=content)
        self.db.add(msg)
        return msg

    def latest(self, session_id: str, limit: int) -> list[ChatMessage]:
        """Ultimele `limit` mesaje (window). DESC + id ca tiebreaker, apoi reversed."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            # id ca tiebreaker: mesajele scrise în aceeași secundă au timestamp egal,
            # iar id-ul auto-increment garantează ordinea reală.
            .order_by(ChatMessage.timestamp.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        rows = self.db.execute(stmt).scalars().all()
        return list(reversed(rows))  # reverse → cronologic pentru LLM


class PersistentMemory:
    """API curat pentru agent. Nu expune SQL/ORM restului aplicației."""

    def __init__(self, window: int = 10):
        self.window = window

    def load_messages(self, session_id: str) -> list[dict]:
        """Ultimele N mesaje (cronologic), gata de injectat în prompt."""
        with transaction() as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            return [{"role": m.role, "content": m.content} for m in rows]

    def save_message(self, session_id: str, role: str, content: str,
                     user_id: str = "default") -> None:
        with transaction() as db:  # tranzacție atomică (unit of work)
            repo = ChatMessageRepository(db)
            repo.ensure_session(session_id, user_id)
            repo.add(session_id, role, content)

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str,
                  user_id: str = "default") -> None:
        """Salvează ambele mesaje (user + assistant) într-o singură tranzacție."""
        with transaction() as db:
            repo = ChatMessageRepository(db)
            repo.ensure_session(session_id, user_id)
            repo.add(session_id, "user", user_msg)
            repo.add(session_id, "assistant", assistant_msg)

    def clear(self, session_id: str) -> None:
        """Șterge conversația unei sesiuni (cascadează la chat_messages)."""
        with transaction() as db:
            sess = db.get(ChatSession, session_id)
            if sess is not None:
                db.delete(sess)
