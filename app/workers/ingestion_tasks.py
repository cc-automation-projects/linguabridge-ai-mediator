from app.core.celery_app import celery_app
from app.core.logger import logger
from app.core.models import IncomingMessage, MediaType


@celery_app.task(
    bind=True,
    name="app.workers.ingestion_tasks.dispatch_to_celery",
    queue="translate_text", # Дефолтная очередь, будет переопределена динамически
    acks_late=True,
    reject_on_worker_lost=True
)
def dispatch_to_celery_task(self, message_dict: dict):
    """
    Celery-задача, которая получает нормализованное сообщение
    и перенаправляет его в специфичную очередь в зависимости от типа медиа.
    """
    try:
        # Восстанавливаем модель из словаря
        msg = IncomingMessage.model_validate(message_dict)

        # Определяем целевую очередь
        target_queue = "translate_voice" if msg.media_type in [MediaType.VOICE, MediaType.VOICE] else "translate_text"

        logger.info(
            "dispatching_to_queue",
            message_id=msg.message_id,
            channel=msg.channel.value,
            target_queue=target_queue,
            has_audio=bool(msg.audio_s3_key)
        )

        # Импортируем локально, чтобы избежать circular imports,
        # и вызываем следующую задачу в правильной очереди
        from app.workers.translation_tasks import process_incoming_message
        process_incoming_message.apply_async(
            args=[message_dict],
            queue=target_queue,
            priority=5 if target_queue == "translate_text" else 3
        )

    except Exception as e:
        logger.error("dispatch_failed", error=str(e), message_dict=message_dict, exc_info=True)
        # Не делаем retry здесь, так как ошибка, скорее всего, в валидации данных.
        # Пусть упадет в DLQ (Dead Letter Queue), если настроена.


# Асинхронная обертка для вызова из FastAPI background tasks
async def dispatch_to_celery(msg: IncomingMessage):
    """Вызывает Celery задачу асинхронно."""
    dispatch_to_celery_task.delay(msg.model_dump())
