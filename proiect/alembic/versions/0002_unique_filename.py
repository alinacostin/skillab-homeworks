"""unique filename pe documents (ingest idempotent)

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-08

Înlocuiește indexul ne-unic `ix_documents_filename` (din 0001) cu o constrângere
UNIQUE, ca re-ingestul aceluiași fișier să nu mai poată duplica documente.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Defensiv: dacă baza are deja duplicate (din rulări anterioare), păstrăm
    # rândul cel mai recent (id maxim) per filename. Chunk-urile cad prin CASCADE.
    op.execute(
        """
        DELETE FROM documents a
        USING documents b
        WHERE a.filename = b.filename
          AND a.id < b.id
        """
    )
    op.drop_index("ix_documents_filename", table_name="documents")
    op.create_unique_constraint("uq_documents_filename", "documents", ["filename"])


def downgrade() -> None:
    op.drop_constraint("uq_documents_filename", "documents", type_="unique")
    op.create_index("ix_documents_filename", "documents", ["filename"])
