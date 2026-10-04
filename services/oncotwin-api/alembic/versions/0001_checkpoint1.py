"""checkpoint 1 durable patient state schema

Revision ID: 0001_checkpoint1
Revises:
"""
from alembic import op

revision = "0001_checkpoint1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Initial migration is intentionally tied to the checkpoint-1 immutable schema.
    # Future checkpoints should ADD migrations rather than editing this file.
    from oncotwin_api.models import Base
    # IMPORTANT: checkpoint migrations must be historically stable.
    # Do not call Base.metadata.create_all() without an explicit table list:
    # Base now contains models added in later checkpoints.
    cp1_table_names = [
        "users",
        "patients",
        "documents",
        "document_pages",
        "extraction_runs",
        "candidate_facts",
        "committed_facts",
        "fact_sources",
        "fact_conflicts",
        "timeline_events",
        "patient_state_snapshots",
        "audit_events",
    ]
    missing = [name for name in cp1_table_names if name not in Base.metadata.tables]
    if missing:
        raise RuntimeError(f"CP1 migration metadata is missing expected tables: {missing}")
    Base.metadata.create_all(
        bind=op.get_bind(),
        tables=[Base.metadata.tables[name] for name in cp1_table_names],
    )
def downgrade() -> None:
    from oncotwin_api.models import Base
    Base.metadata.drop_all(bind=op.get_bind())
