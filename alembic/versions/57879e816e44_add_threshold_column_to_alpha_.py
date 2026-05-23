"""add threshold column to alpha_acceptance_daily

Revision ID: 57879e816e44
Revises: 7daf452b3a1e
Create Date: 2026-05-23 13:07:36.391821

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '57879e816e44'
down_revision: Union[str, Sequence[str], None] = '7daf452b3a1e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "alpha_acceptance_daily",
        sa.Column("threshold", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("alpha_acceptance_daily", "threshold")
