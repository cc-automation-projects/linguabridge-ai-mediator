from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "linguabridge",
    broker=settings.rabbitmq_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="translate_text",
    task_queues={
        "translate_text": {"exchange": "linguabridge", "routing_key": "translate.text"},
        "translate_voice": {"exchange": "linguabridge", "routing_key": "translate.voice"},
        "tts_generation": {"exchange": "linguabridge", "routing_key": "tts.generate"},
    },
    task_default_exchange="linguabridge",
    task_default_routing_key="translate.text",
)
