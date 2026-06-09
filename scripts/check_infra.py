"""Скрипт проверки работоспособности всей инфраструктуры."""
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import close_db, init_db
from app.core.logger import logger, setup_logging
from app.core.rabbitmq import rabbitmq_client
from app.core.redis import redis_client
from app.core.s3 import s3_client


async def check_postgres() -> bool:
    try:
        await init_db()
        logger.info("✅ PostgreSQL: OK")
        return True
    except Exception as e:
        logger.error("❌ PostgreSQL: FAILED", error=str(e))
        return False


async def check_redis() -> bool:
    try:
        await redis_client.connect()
        await redis_client.client.set("linguabridge:health:check", "1", ex=10)
        value = await redis_client.client.get("linguabridge:health:check")
        assert value == "1"
        logger.info("✅ Redis: OK")
        return True
    except Exception as e:
        logger.error("❌ Redis: FAILED", error=str(e))
        return False


async def check_rabbitmq() -> bool:
    try:
        await rabbitmq_client.connect()
        text_queue = rabbitmq_client.get_queue("text")
        voice_queue = rabbitmq_client.get_queue("voice")
        dlq = rabbitmq_client.get_queue("dlq")
        assert text_queue is not None
        assert voice_queue is not None
        assert dlq is not None
        logger.info(
            "✅ RabbitMQ: OK",
            text_queue=settings.rabbitmq_queue_text,
            voice_queue=settings.rabbitmq_queue_voice,
            dlq=settings.rabbitmq_queue_dlq,
        )
        return True
    except Exception as e:
        logger.error("❌ RabbitMQ: FAILED", error=str(e))
        return False


async def check_minio() -> bool:
    try:
        await s3_client.ensure_bucket_exists()
        await s3_client.configure_lifecycle()

        # Тестовая загрузка
        test_key = "health-check/test.txt"
        await s3_client.upload_file(
            key=test_key,
            body=b"linguabridge health check",
            content_type="text/plain",
        )

        # Тестовое скачивание
        data = await s3_client.download_file(test_key)
        assert data == b"linguabridge health check"

        # Очистка
        await s3_client.delete_file(test_key)

        logger.info("✅ MinIO: OK", bucket=settings.minio_bucket)
        return True
    except Exception as e:
        logger.error("❌ MinIO: FAILED", error=str(e))
        return False


async def main() -> int:
    setup_logging()
    logger.info("=== LinguaBridge Infrastructure Check ===")

    results = await asyncio.gather(
        check_postgres(),
        check_redis(),
        check_rabbitmq(),
        check_minio(),
    )

    await redis_client.close()
    await rabbitmq_client.close()
    await close_db()

    if all(results):
        logger.info("=== All checks passed ===")
        return 0
    else:
        logger.error("=== Some checks failed ===")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
