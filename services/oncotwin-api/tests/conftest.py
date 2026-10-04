import os
from pathlib import Path

import pytest


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    storage = tmp_path / "storage"
    monkeypatch.setenv("ONCOTWIN_ENVIRONMENT", "test")
    monkeypatch.setenv("ONCOTWIN_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ONCOTWIN_STORAGE_BACKEND", "local")
    monkeypatch.setenv("ONCOTWIN_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("ONCOTWIN_JWT_SECRET", "test-secret")
    monkeypatch.setenv("ONCOTWIN_TASKS_EAGER", "true")
    monkeypatch.setenv("ONCOTWIN_EXTRACTOR_PROVIDER", "rules")
    monkeypatch.setenv("ONCOTWIN_AUTO_CREATE_SCHEMA", "true")
    monkeypatch.setenv("ONCOTWIN_DEMO_MODE", "true")

    from oncotwin_api.config import get_settings
    from oncotwin_api.db import reset_db_caches
    from oncotwin_api.storage import get_object_store

    get_settings.cache_clear()
    reset_db_caches()
    get_object_store.cache_clear()
    yield tmp_path
    get_object_store.cache_clear()
    reset_db_caches()
    get_settings.cache_clear()


@pytest.fixture()
def client(app_env):
    from fastapi.testclient import TestClient
    from oncotwin_api.main import app
    with TestClient(app) as c:
        yield c
