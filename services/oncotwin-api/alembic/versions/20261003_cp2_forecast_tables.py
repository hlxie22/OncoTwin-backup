"""CP2 frozen forecast persistence tables.

Revision ID: 20261003_cp2_forecast
Revises: 0001_checkpoint1
"""
from alembic import op
import sqlalchemy as sa

revision = "20261003_cp2_forecast"
down_revision = '0001_checkpoint1'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "support_assessments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reason_codes_json", sa.Text(), nullable=False),
        sa.Column("explanations_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(48), nullable=False),
    )
    op.create_index("ix_support_assessments_patient", "support_assessments", ["patient_id", "created_at"])

    op.create_table(
        "model_input_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
        sa.Column("state_snapshot_id", sa.String(36), sa.ForeignKey("patient_state_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("state_hash", sa.String(64), nullable=False),
        sa.Column("current_scan_fact_id", sa.String(36), nullable=True),
        sa.Column("adapter_version", sa.String(80), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("input_metadata_json", sa.Text(), nullable=False),
        sa.Column("prepared_hashes_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(48), nullable=False),
    )
    op.create_index("ix_model_input_patient", "model_input_snapshots", ["patient_id", "created_at"])
    op.create_index("ix_model_input_request_hash", "model_input_snapshots", ["request_hash"])

    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(36), sa.ForeignKey("patients.id", ondelete="CASCADE"), nullable=False),
        sa.Column("state_snapshot_id", sa.String(36), sa.ForeignKey("patient_state_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("state_hash", sa.String(64), nullable=False),
        sa.Column("model_input_snapshot_id", sa.String(36), sa.ForeignKey("model_input_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("support_assessment_id", sa.String(36), sa.ForeignKey("support_assessments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("current_scan_fact_id", sa.String(36), nullable=True),
        sa.Column("run_status", sa.String(24), nullable=False),
        sa.Column("checkpoint", sa.String(32), nullable=False),
        sa.Column("selected_rule", sa.String(100), nullable=False),
        sa.Column("locked_alpha", sa.Float(), nullable=False),
        sa.Column("deploy_sha256", sa.String(64), nullable=False),
        sa.Column("temporal_encoder_sha256", sa.String(64), nullable=False),
        sa.Column("research_status", sa.String(80), nullable=False),
        sa.Column("pre_logits_json", sa.Text(), nullable=True),
        sa.Column("post_logits_json", sa.Text(), nullable=True),
        sa.Column("selected_logits_json", sa.Text(), nullable=True),
        sa.Column("curves_json", sa.Text(), nullable=True),
        sa.Column("horizons_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(48), nullable=False),
    )
    op.create_index("ix_forecast_patient", "forecast_runs", ["patient_id", "created_at"])
    op.create_index("ix_forecast_state_hash", "forecast_runs", ["state_hash"])


def downgrade():
    op.drop_index("ix_forecast_state_hash", table_name="forecast_runs")
    op.drop_index("ix_forecast_patient", table_name="forecast_runs")
    op.drop_table("forecast_runs")
    op.drop_index("ix_model_input_request_hash", table_name="model_input_snapshots")
    op.drop_index("ix_model_input_patient", table_name="model_input_snapshots")
    op.drop_table("model_input_snapshots")
    op.drop_index("ix_support_assessments_patient", table_name="support_assessments")
    op.drop_table("support_assessments")
