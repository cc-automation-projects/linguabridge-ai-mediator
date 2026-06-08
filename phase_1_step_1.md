# ЭТАП 1, ПОДЗАДАЧА 1.1: Развертывание базовой инфраструктуры

## Шаг 1.1.1: Структура проекта и зависимости

Создадим профессиональную структуру Python-проекта с разделением ответственности.

**Выполните команды:**
```bash
mkdir linguabridge && cd linguabridge
mkdir -p app/{core,adapters,models,services,workers,utils}
mkdir -p app/adapters/{max_adapter,vk_adapter,telegram_adapter}
mkdir -p alembic/versions
mkdir -p tests/{unit,integration}
touch app/__init__.py app/core/__init__.py app/adapters/__init__.py \
      app/models/__init__.py app/services/__init__.py app/workers/__init__.py \
      app/utils/__init__.py
```

**Файл: `pyproject.toml`**
```toml
[tool.poetry]
name = "linguabridge-ai-mediator"
version = "0.1.0"
description = "Омниканальный AI-медиатор для мгновенного двунаправленного перевода диалогов"
authors = ["Your Team <dev@company.com>"]
package-mode = false

[tool.poetry.dependencies]
python = "^3.12"

# === Web Framework ===
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.32.0"}

# === Async HTTP Client ===
httpx = "^0.27.2"

# === Bot Frameworks ===
aiogram = "^3.13.0"          # Telegram
vk-api = "^11.9.9"           # VK

# === Task Queue ===
celery = {extras = ["redis"], version = "^5.4.0"}
redis = "^5.2.0"

# === Database ===
sqlalchemy = {extras = ["asyncio"], version = "^2.0.35"}
asyncpg = "^0.30.0"
alembic = "^1.13.3"

# === S3 Storage ===
aiobotocore = "^2.15.2"

# === Validation & Config ===
pydantic = "^2.9.2"
pydantic-settings = "^2.5.2"

# === ML/NLP ===
fasttext = "^0.9.3"
transformers = "^4.46.0"
torch = "^2.5.0"
bitsandbytes = "^0.44.1"
faster-whisper = "^1.0.3"
silero = "^0.4.1"

# === PII Masking ===
presidio-analyzer = "^2.2.35"
presidio-anonymizer = "^2.2.35"
spacy = "^3.7.5"

# === Audio Processing ===
ffmpeg-python = "^0.2.0"
soundfile = "^0.12.1"

# === Observability ===
structlog = "^24.4.0"
opentelemetry-api = "^1.27.0"
opentelemetry-sdk = "^1.27.0"
opentelemetry-exporter-otlp = "^1.27.0"
opentelemetry-instrumentation-fastapi = "^0.48b0"
opentelemetry-instrumentation-celery = "^0.48b0"

# === Circuit Breaker & Retry ===
pybreaker = "^1.1.0"
tenacity = "^9.0.0"

# === Rate Limiting ===
slowapi = "^0.1.9"
limits = "^3.13.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"
pytest-asyncio = "^0.24.0"
pytest-cov = "^5.0.0"
ruff = "^0.7.0"
mypy = "^1.11.2"
httpx = "^0.27.2"  # для TestClient

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 100
target-version = "py312"
select = ["E", "F", "I", "N", "W", "B", "UP", "RUF", "ASYNC"]

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

*Действие:* `poetry install`

---

## Шаг 1.1.2: Строгая конфигурация (Pydantic Settings)

Централизованное управление всеми настройками с валидацией при старте.

**Файл: `app/core/config.py`**
```python
from functools import lru_cache
from typing import Literal
from pydantic import Field, PostgresDsn, RedisDsn, AnyUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Централизованная конфигурация приложения."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # === Application ===
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    app_name: str = Field(default="LinguaBridge AI Mediator")
    app_version: str = Field(default="0.1.0")
    app_debug: bool = Field(default=False)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)

    # === PostgreSQL ===
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="linguabridge")
    postgres_password: SecretStr = Field(default=SecretStr("linguabridge"))
    postgres_db: str = Field(default="linguabridge")

    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_dsn_sync(self) -> str:
        """Для Alembic миграций (синхронный драйвер)."""
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # === Redis ===
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: SecretStr = Field(default=SecretStr(""))
    redis_db: int = Field(default=0)

    @property
    def redis_url(self) -> str:
        pwd = self.redis_password.get_secret_value()
        auth = f":{pwd}@" if pwd else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # === RabbitMQ ===
    rabbitmq_host: str = Field(default="localhost")
    rabbitmq_port: int = Field(default=5672)
    rabbitmq_user: str = Field(default="guest")
    rabbitmq_password: SecretStr = Field(default=SecretStr("guest"))
    rabbitmq_vhost: str = Field(default="/")
    
    # Очереди
    rabbitmq_queue_text: str = Field(default="linguabridge.translate.text")
    rabbitmq_queue_voice: str = Field(default="linguabridge.translate.voice")
    rabbitmq_queue_dlq: str = Field(default="linguabridge.translate.dlq")
    rabbitmq_exchange: str = Field(default="linguabridge.exchange")

    @property
    def rabbitmq_url(self) -> str:
        pwd = self.rabbitmq_password.get_secret_value()
        return f"amqp://{self.rabbitmq_user}:{pwd}@{self.rabbitmq_host}:{self.rabbitmq_port}/{self.rabbitmq_vhost}"

    # === MinIO (S3-compatible) ===
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: SecretStr = Field(default=SecretStr("minioadmin"))
    minio_bucket: str = Field(default="linguabridge-media")
    minio_region: str = Field(default="us-east-1")
    minio_audio_retention_days: int = Field(default=7, description="Lifecycle: удаление аудио через N дней (152-ФЗ)")

    # === Channel: MAX ===
    max_bot_token: SecretStr = Field(default=SecretStr(""))
    max_webhook_secret: SecretStr = Field(default=SecretStr(""))
    max_api_base_url: AnyUrl = Field(default="https://botapi.max.ru")  # Актуальный URL из dev.max.ru
    max_rate_limit_per_second: int = Field(default=30)

    # === Channel: VK ===
    vk_bot_token: SecretStr = Field(default=SecretStr(""))
    vk_confirmation_token: SecretStr = Field(default=SecretStr(""))
    vk_api_version: str = Field(default="5.199")

    # === Channel: Telegram ===
    telegram_bot_token: SecretStr = Field(default=SecretStr(""))
    telegram_webhook_secret: SecretStr = Field(default=SecretStr(""))

    # === ML Models ===
    fasttext_model_path: str = Field(default="./models/lid.176.bin")
    nllb_model_name: str = Field(default="facebook/nllb-200-distilled-600M")
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v3-turbo"] = Field(
        default="medium"
    )
    whisper_compute_type: str = Field(default="int8")

    # === Translation Settings ===
    supported_languages: list[str] = Field(
        default=["tg", "uz", "ky", "zh", "en", "ru"],
        description="ISO 639-1 codes supported languages"
    )
    translation_latency_target_ms: int = Field(default=1500)
    voice_latency_target_ms: int = Field(default=4000)

    # === Observability ===
    otlp_endpoint: str = Field(default="http://localhost:4317")
    otlp_service_name: str = Field(default="linguabridge")


@lru_cache
def get_settings() -> Settings:
    """Кэшированный доступ к конфигурации."""
    return Settings()


settings = get_settings()
```

**Файл: `.env.example`**
```env
APP_ENV=development
APP_DEBUG=true

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=linguabridge
POSTGRES_PASSWORD=linguabridge
POSTGRES_DB=linguabridge

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=linguabridge-media

MAX_BOT_TOKEN=your_max_bot_token
MAX_WEBHOOK_SECRET=your_max_secret
VK_BOT_TOKEN=your_vk_bot_token
VK_CONFIRMATION_TOKEN=your_vk_confirmation
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

---

## Шаг 1.1.3: Инфраструктура как код (Docker Compose)

Локальное окружение для разработки со всеми необходимыми сервисами.

**Файл: `docker-compose.yml`**
```yaml
version: '3.9'

services:
  # === PostgreSQL 15 ===
  postgres:
    image: postgres:15-alpine
    container_name: linguabridge-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: linguabridge
      POSTGRES_PASSWORD: linguabridge
      POSTGRES_DB: linguabridge
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U linguabridge"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - linguabridge-net

  # === Redis 7 with AOF persistence ===
  redis:
    image: redis:7-alpine
    container_name: linguabridge-redis
    restart: unless-stopped
    command: >
      redis-server 
      --appendonly yes 
      --maxmemory 512mb 
      --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - linguabridge-net

  # === RabbitMQ 3.13 with Management UI ===
  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    container_name: linguabridge-rabbitmq
    restart: unless-stopped
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
      RABBITMQ_DEFAULT_VHOST: /
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"]
      interval: 15s
      timeout: 10s
      retries: 5
    networks:
      - linguabridge-net

  # === MinIO (S3-compatible) ===
  minio:
    image: minio/minio:latest
    container_name: linguabridge-minio
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Console UI
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - linguabridge-net

  # === MinIO bucket & lifecycle initialization ===
  minio-init:
    image: minio/mc:latest
    container_name: linguabridge-minio-init
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      sleep 5;
      mc alias set linguabridge http://minio:9000 minioadmin minioadmin;
      mc mb --ignore-existing linguabridge/linguabridge-media;
      mc anonymous set download linguabridge/linguabridge-media;
      echo 'Bucket created successfully';
      "
    networks:
      - linguabridge-net

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  minio_data:

networks:
  linguabridge-net:
    driver: bridge
```

*Действие:* `docker compose up -d` и дождаться `healthy` статуса всех сервисов (`docker compose ps`).

---

## Шаг 1.1.4: Асинхронный слой доступа к PostgreSQL (SQLAlchemy)

Создадим единый механизм работы с БД с пулом соединений и graceful shutdown.

**Файл: `app/core/database.py`**
```python
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""
    pass


def _create_engine() -> AsyncEngine:
    """Создание движка с оптимальными параметрами пула соединений."""
    return create_async_engine(
        settings.postgres_dsn,
        echo=settings.app_debug,
        pool_size=20,           # Базовый размер пула
        max_overflow=10,        # Максимум дополнительных соединений
        pool_timeout=30,        # Таймаут ожидания соединения (сек)
        pool_recycle=3600,      # Переподключение каждый час
        pool_pre_ping=True,     # Проверка "живости" соединения перед использованием
    )


engine: AsyncEngine = _create_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Зависимость для FastAPI endpoint-ов.
    Автоматически закрывает сессию и откатывает транзакцию при ошибке.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Создание таблиц (для dev/test окружения)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Корректное закрытие пула соединений."""
    await engine.dispose()
```

---

## Шаг 1.1.5: Клиент Redis для кэширования и контекста

**Файл: `app/core/redis.py`**
```python
from typing import Optional
import redis.asyncio as redis
from app.core.config import settings


class RedisClient:
    """Обертка над redis.asyncio с оптимальными настройками."""

    def __init__(self) -> None:
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Создание пула соединений."""
        self._pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Проверка соединения
        await self._client.ping()

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()


redis_client = RedisClient()
```

---

## Шаг 1.1.6: Клиент MinIO (S3) с Lifecycle политикой

**Файл: `app/core/s3.py`**
```python
import json
from typing import Optional
from aiobotocore.session import get_session
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError
from app.core.config import settings


class S3Client:
    """Асинхронный клиент для S3-совместимого хранилища (MinIO)."""

    def __init__(self) -> None:
        self._session = get_session()
        self._config = AioConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=30,
        )

    def _client_context(self):
        return self._session.create_client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key.get_secret_value(),
            region_name=settings.minio_region,
            config=self._config,
        )

    async def ensure_bucket_exists(self) -> None:
        """Создание бакета, если он не существует."""
        async with self._client_context() as client:
            try:
                await client.head_bucket(Bucket=settings.minio_bucket)
            except ClientError:
                await client.create_bucket(Bucket=settings.minio_bucket)

    async def configure_lifecycle(self) -> None:
        """
        Настройка автоматического удаления объектов через N дней (152-ФЗ).
        """
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "auto-delete-old-audio",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},  # Применяется ко всем объектам
                    "Expiration": {"Days": settings.minio_audio_retention_days},
                }
            ]
        }
        async with self._client_context() as client:
            try:
                await client.put_bucket_lifecycle_configuration(
                    Bucket=settings.minio_bucket,
                    LifecycleConfiguration=lifecycle_config,
                )
            except ClientError as e:
                # Некоторые S3-совместимые хранилища не поддерживают lifecycle
                # Логируем warning, но не падаем
                print(f"Warning: Lifecycle policy not applied: {e}")

    async def upload_file(
        self,
        key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        """Загрузка файла в бакет."""
        async with self._client_context() as client:
            await client.put_object(
                Bucket=settings.minio_bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        return key

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Генерация временной ссылки для скачивания."""
        async with self._client_context() as client:
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.minio_bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url

    async def download_file(self, key: str) -> bytes:
        """Скачивание файла из бакета."""
        async with self._client_context() as client:
            response = await client.get_object(
                Bucket=settings.minio_bucket,
                Key=key,
            )
            async with response["Body"] as stream:
                return await stream.read()

    async def delete_file(self, key: str) -> None:
        """Удаление файла."""
        async with self._client_context() as client:
            await client.delete_object(Bucket=settings.minio_bucket, Key=key)


s3_client = S3Client()
```

---

## Шаг 1.1.7: Клиент RabbitMQ и декларация очередей

**Файл: `app/core/rabbitmq.py`**
```python
from typing import Optional
import aio_pika
from aio_pika.abc import AbstractRobustConnection, AbstractChannel, AbstractQueue, AbstractExchange
from app.core.config import settings


class RabbitMQClient:
    """Асинхронный клиент RabbitMQ с robust connection."""

    def __init__(self) -> None:
        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._exchange: Optional[AbstractExchange] = None
        self._queues: dict[str, AbstractQueue] = {}

    async def connect(self) -> None:
        """Установка соединения и декларация инфраструктуры очередей."""
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        
        # QoS: каждый воркер берет не более 1 сообщения
        await self._channel.set_qos(prefetch_count=1)

        # Декларируем exchange (topic type для гибкой маршрутизации)
        self._exchange = await self._channel.declare_exchange(
            settings.rabbitmq_exchange,
            type=aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

        # Декларируем DLQ (Dead Letter Queue)
        dlq = await self._channel.declare_queue(
            settings.rabbitmq_queue_dlq,
            durable=True,
        )
        self._queues["dlq"] = dlq

        # Декларируем основную очередь для текстовых сообщений (высокий приоритет)
        text_queue = await self._channel.declare_queue(
            settings.rabbitmq_queue_text,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": settings.rabbitmq_queue_dlq,
                "x-max-priority": 10,  # Поддержка приоритетов
            },
        )
        await text_queue.bind(self._exchange, routing_key="translate.text")
        self._queues["text"] = text_queue

        # Декларируем очередь для голосовых сообщений (низкий приоритет, тяжелее)
        voice_queue = await self._channel.declare_queue(
            settings.rabbitmq_queue_voice,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": settings.rabbitmq_queue_dlq,
                "x-max-priority": 5,
            },
        )
        await voice_queue.bind(self._exchange, routing_key="translate.voice")
        self._queues["voice"] = voice_queue

    @property
    def channel(self) -> AbstractChannel:
        if self._channel is None:
            raise RuntimeError("RabbitMQ not connected")
        return self._channel

    @property
    def exchange(self) -> AbstractExchange:
        if self._exchange is None:
            raise RuntimeError("RabbitMQ not connected")
        return self._exchange

    def get_queue(self, name: str) -> AbstractQueue:
        if name not in self._queues:
            raise KeyError(f"Queue '{name}' not found")
        return self._queues[name]

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


rabbitmq_client = RabbitMQClient()
```

---

## Шаг 1.1.8: Структурированное логирование (structlog)

**Файл: `app/core/logger.py`**
```python
import logging
import sys
from contextvars import ContextVar
import structlog
from app.core.config import settings

# Context variable для сквозной трассировки (пробрасывается через Celery/async)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
channel_var: ContextVar[str | None] = ContextVar("channel", default=None)


def _add_context(logger, method_name, event_dict):
    """Добавление контекстных переменных в каждый лог."""
    if trace_id := trace_id_var.get():
        event_dict["trace_id"] = trace_id
    if channel := channel_var.get():
        event_dict["channel"] = channel
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """Инициализация структурированного логирования."""
    log_level = logging.DEBUG if settings.app_debug else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.app_env == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True, sort_keys=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("linguabridge")
```

---

## Шаг 1.1.9: Точка входа FastAPI (Webhook Server)

**Файл: `app/main.py`**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logger import setup_logging, logger
from app.core.database import init_db, close_db
from app.core.redis import redis_client
from app.core.rabbitmq import rabbitmq_client
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
```

---

## Шаг 1.1.10: Скрипт проверки инфраструктуры

Создадим скрипт, который проверит работоспособность всех компонентов.

**Файл: `scripts/check_infra.py`**
```python
"""Скрипт проверки работоспособности всей инфраструктуры."""
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logger import setup_logging, logger
from app.core.database import init_db, close_db
from app.core.redis import redis_client
from app.core.rabbitmq import rabbitmq_client
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
```

---

## Шаг 1.1.11: Запуск и проверка

**1. Убедитесь, что инфраструктура запущена:**
```bash
docker compose up -d
docker compose ps
```
Все сервисы должны иметь статус `healthy`.

**2. Создайте `.env` файл:**
```bash
cp .env.example .env
```

**3. Запустите проверку инфраструктуры:**
```bash
poetry run python scripts/check_infra.py
```

**Ожидаемый результат:**
```
=== LinguaBridge Infrastructure Check ===
✅ PostgreSQL: OK
✅ Redis: OK
✅ RabbitMQ: OK  text_queue=linguabridge.translate.text  voice_queue=linguabridge.translate.voice  dlq=linguabridge.translate.dlq
✅ MinIO: OK  bucket=linguabridge-media
=== All checks passed ===
```

**4. Запустите FastAPI сервер:**
```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**5. Проверьте endpoints:**
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`
- Swagger UI (в dev режиме): `http://localhost:8000/docs`

**6. Дополнительно — проверьте UI сервисов:**
- RabbitMQ Management: `http://localhost:15672` (guest/guest)
- MinIO Console: `http://localhost:9001` (minioadmin/minioadmin)

Вы должны увидеть:
- 3 очереди в RabbitMQ (`linguabridge.translate.text`, `linguabridge.translate.voice`, `linguabridge.translate.dlq`)
- Бакет `linguabridge-media` в MinIO

---

## Что мы достигли в Подзадаче 1.1

✅ **Профессиональная структура проекта:** Разделение на слои (core, adapters, models, services, workers, utils) с четкими границами ответственности.

✅ **Production-grade конфигурация:** Pydantic Settings с валидацией, секретными полями (`SecretStr`), вычисляемыми свойствами (DSN) и поддержкой вложенных переменных окружения.

✅ **Полноценная инфраструктура:** Docker Compose со всеми необходимыми сервисами, healthcheck-ами, persistence и правильными параметрами (Redis AOF + maxmemory, RabbitMQ Management, MinIO с Console UI).

✅ **Lifecycle-политика для 152-ФЗ:** Автоматическое удаление аудиофайлов из S3 через 7 дней, настроенное программно при старте приложения.

✅ **Асинхронный стек:** Все клиенты (PostgreSQL, Redis, S3, RabbitMQ) полностью асинхронные, с пулами соединений, таймаутами и retry-логикой.

✅ **Очереди с приоритетами:** Разделение текстовых (высокий приоритет) и голосовых (низкий приоритет) задач + DLQ для обработки сбоев.

✅ **Структурированное логирование:** structlog с context variables для сквозной трассировки (trace_id, channel), JSON-рендерингом в production.

✅ **Graceful shutdown:** Корректное закрытие всех соединений при остановке приложения через FastAPI lifespan.

✅ **Автоматизированная проверка:** Скрипт `check_infra.py` позволяет быстро диагностировать состояние всех компонентов.
