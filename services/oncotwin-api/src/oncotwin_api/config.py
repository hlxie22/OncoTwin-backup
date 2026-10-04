from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONCOTWIN_", env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///./.oncotwin-dev/oncotwin.db"
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: Path = Path("./.oncotwin-dev/storage")

    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    jwt_secret: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    jwt_ttl_hours: int = 24

    tasks_eager: bool = True
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"

    extractor_provider: Literal["rules", "openai", "router"] = "rules"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6"


    # Checkpoint 4 shared language-intelligence layer. Model IDs are configuration,
    # not application logic because provider catalogs and free-tier quotas can change.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_standard_model: str = "openai/gpt-oss-120b"
    groq_qwen_model: str = "qwen/qwen3.8-27b"
    groq_cheap_model: str = "openai/gpt-oss-20b"
    google_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_hard_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 45.0
    llm_rate_limit_cooldown_seconds: int = 900
    evidence_mode: Literal["live", "snapshot"] = "live"
    evidence_timeout_seconds: float = 25.0
    ncbi_email: str | None = None
    extraction_confidence_threshold: float = 0.90
    max_document_chars: int = 120_000

    auto_create_schema: bool = False
    demo_mode: bool = True
    allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite:")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
