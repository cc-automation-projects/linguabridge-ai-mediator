from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.adapters.max_adapter import router as max_router
from app.adapters.telegram_adapter import router as telegram_router
from app.adapters.vk_adapter import router as vk_router
from app.core.config import settings
from app.core.database import close_db, init_db
from app.core.logger import logger, setup_logging
from app.core.rabbitmq import rabbitmq_client
from app.core.redis import redis_client
from app.core.s3 import s3_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    setup_logging()
    logger.info("application_starting", version=settings.app_version)

    # === Startup ===
    try:
        await init_db()
        logger.info("database_initialized")

        await redis_client.connect()
        logger.info("redis_connected")

        await rabbitmq_client.connect()
        logger.info("rabbitmq_connected_queues_declared")

        await s3_client.ensure_bucket_exists()
        await s3_client.configure_lifecycle()
        logger.info("s3_bucket_ready", bucket=settings.minio_bucket)

        logger.info("application_ready", host=settings.app_host, port=settings.app_port)
        yield

    except Exception as e:
        logger.critical("startup_failed", error=str(e), exc_info=True)
        raise

    finally:
        # === Shutdown ===
        logger.info("application_shutting_down")
        await redis_client.close()
        await rabbitmq_client.close()
        await close_db()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Омниканальный AI-медиатор для мгновенного перевода диалогов",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_debug else None,
    redoc_url="/redoc" if settings.app_debug else None,
)

# Подключение роутеров каналов
app.include_router(max_router, prefix="/webhooks")
app.include_router(vk_router, prefix="/webhooks")
app.include_router(telegram_router, prefix="/webhooks")


@app.get("/health", tags=["system"])
async def health_check():
    """Базовая проверка работоспособности."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "env": settings.app_env,
    }


@app.get("/ready", tags=["system"])
async def readiness_check():
    """Глубокая проверка готовности всех зависимостей."""
    checks = {}
    try:
        await redis_client.client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        if rabbitmq_client.channel is not None:
            checks["rabbitmq"] = "ok"
    except Exception as e:
        checks["rabbitmq"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений для чистых логов."""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
