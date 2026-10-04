from .tasks import celery_app

if celery_app is None:
    raise RuntimeError("Celery is not installed; install checkpoint-1 dependencies and set ONCOTWIN_TASKS_EAGER=false for a worker")
