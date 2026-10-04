"""CP4 intelligence artifacts, cache provenance, and provider cooldowns.

Revision ID: 20261004_cp4_intelligence
Revises: 20261003_cp2_forecast
"""
from alembic import op
import sqlalchemy as sa

revision = "20261004_cp4_intelligence"
down_revision = "20261003_cp2_forecast"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provider_cooldowns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("model_name", sa.String(200), nullable=False),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("provider", "model_name", name="uq_provider_cooldown_model"),
    )
    op.create_index("ix_provider_cooldowns_until", "provider_cooldowns", ["cooldown_until"])

    op.create_table(
        "generated_artifacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("artifact_type", sa.String(120), nullable=False),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("provider", sa.String(80), nullable=True),
        sa.Column("model_name", sa.String(200), nullable=True),
        sa.Column("prompt_version", sa.String(100), nullable=False),
        sa.Column("schema_version", sa.String(100), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("state_hash", sa.String(64), nullable=True),
        sa.Column("source_hashes_json", sa.JSON(), nullable=False),
        sa.Column("content_json", sa.JSON(), nullable=False),
        sa.Column("grounding_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("cache_key", name="uq_generated_artifact_cache_key"),
    )
    op.create_index("ix_generated_artifact_patient_type", "generated_artifacts", ["patient_id", "artifact_type", "created_at"])
    op.create_index("ix_generated_artifact_document", "generated_artifacts", ["document_id", "artifact_type", "created_at"])
    op.create_index("ix_generated_artifact_state", "generated_artifacts", ["state_hash"])


def downgrade() -> None:
    op.drop_index("ix_generated_artifact_state", table_name="generated_artifacts")
    op.drop_index("ix_generated_artifact_document", table_name="generated_artifacts")
    op.drop_index("ix_generated_artifact_patient_type", table_name="generated_artifacts")
    op.drop_table("generated_artifacts")
    op.drop_index("ix_provider_cooldowns_until", table_name="provider_cooldowns")
    op.drop_table("provider_cooldowns")
