from __future__ import annotations

from .config import get_settings
from .db import get_sessionmaker
from .processing import process_document


def _run(document_id: str, user_id: str) -> None:
    db = get_sessionmaker()()
    try:
        process_document(db, document_id, user_id)
    finally:
        db.close()


try:
    from celery import Celery
except ImportError:  # allows eager cluster preview even before optional worker deps are installed
    Celery = None

settings = get_settings()

if Celery is not None:
    celery_app = Celery(
        "oncotwin",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    celery_app.conf.update(
        task_always_eager=settings.tasks_eager,
        task_eager_propagates=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )

    @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
    def process_document_task(self, document_id: str, user_id: str) -> None:
        _run(document_id, user_id)
else:
    if not settings.tasks_eager:
        raise RuntimeError("Celery is required when ONCOTWIN_TASKS_EAGER=false")

    class _EagerTask:
        def delay(self, document_id: str, user_id: str) -> None:
            _run(document_id, user_id)

    process_document_task = _EagerTask()
    celery_app = None
