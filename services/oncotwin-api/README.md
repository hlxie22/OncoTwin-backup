# OncoTwin API - Checkpoint 1

Final-path patient-state/ingestion service for the CAC/research app.

- SQLAlchemy schema supports PostgreSQL; SQLite is supported for Slurm preview/development.
- Object storage supports local filesystem or S3-compatible services.
- Celery is the task boundary; `ONCOTWIN_TASKS_EAGER=true` executes the exact task locally for cluster preview.
- Extraction supports a deterministic rules provider for golden tests and an OpenAI Structured Outputs provider for real source-grounded LLM extraction.
- Quantitative forecasting is intentionally not in checkpoint 1; checkpoint 2 will consume committed state only.

Do not place real PHI in the CAC/development deployment.
