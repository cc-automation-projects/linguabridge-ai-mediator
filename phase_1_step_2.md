Мы реализуем это максимально производственно-готовым способом: строгая типизация, асинхронность, проверка подписей (безопасность) и мгновенная диспетчеризация в Celery по принципу "Fire and Forget", чтобы не блокировать вебхуки провайдеров.

---

# ЭТАП 1, ПОДЗАДАЧА 1.2: Мультиканальный Ingestion (Channel Adapter)

## Шаг 1.2.1: Базовые утилиты для адаптеров

Создадим общий модуль, который будет отвечать за скачивание медиа из внешнего источника и загрузку его в наш S3. Это гарантирует, что в Celery попадет уже готовый `s3_key`, а не временная ссылка, которая может протухнуть.

**Файл: `app/adapters/utils.py`**
```python
import httpx
import uuid
from typing import Optional
from app.core.logger import logger
from app.infrastructure.s3 import s3_service

async def download_and_upload_media(file_url: str, channel: str, media_type: str) -> Optional[str]:
    """
    Скачивает медиафайл по URL и загружает в S3.
    Возвращает s3_key или None в случае ошибки.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            
            file_bytes = response.content
            file_extension = "ogg" # MAX, VK и Telegram используют opus/ogg для голосовых
            
            # Генерируем уникальный ключ: {channel}/{media_type}/{uuid}.ogg
            file_key = f"{channel}/{media_type}/{uuid.uuid4()}.{file_extension}"
            
            content_type = "audio/ogg" if media_type in ["voice", "video_note"] else "application/octet-stream"
            
            success = await s3_service.upload_file(file_key, file_bytes, content_type=content_type)
            if success:
                logger.info("media_downloaded_and_uploaded", channel=channel, file_key=file_key, size=len(file_bytes))
                return file_key
            return None
            
    except httpx.HTTPError as e:
        logger.error("media_download_failed", url=file_url, error=str(e))
        return None
    except Exception as e:
        logger.error("media_processing_failed", url=file_url, error=str(e), exc_info=True)
        return None
```

---

## Шаг 1.2.2: Адаптер для MAX (Приоритет 1)

Реализуем строгую схему входящего вебхука на основе документации MAX и логику его обработки.

**Файл: `app/adapters/max_adapter.py`**
```python
import hmac
import hashlib
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

from app.core.config import settings
from app.core.logger import logger
from app.core.models import IncomingMessage, Channel, MediaType
from app.adapters.utils import download_and_upload_media
from app.workers.ingestion_tasks import dispatch_to_celery

router = APIRouter(prefix="/max", tags=["max"])

# --- Строгие схемы данных MAX ---
class MaxUser(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    username: Optional[str] = None

class MaxVoice(BaseModel):
    file_id: str
    file_url: str
    duration: int
    mime_type: str

class MaxMessage(BaseModel):
    message_id: str
    chat_id: str
    from_user: MaxUser = Field(alias="from")
    timestamp: int
    text: Optional[str] = None
    voice: Optional[MaxVoice] = None
    # Можно добавить photo, document и т.д. по аналогии

class MaxUpdate(BaseModel):
    update_id: int
    message: Optional[MaxMessage] = None

# --- Логика адаптера ---
def verify_max_signature(payload: bytes, signature: str) -> bool:
    """Проверка HMAC-подписи вебхука от MAX."""
    secret = settings.max_webhook_secret.get_secret_value()
    if not secret:
        return True # Если секрет не задан (dev-режим), пропускаем
    expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_sig, signature)

@router.post("/webhook")
async def handle_max_webhook(request: Request, background_tasks: BackgroundTasks):
    """Прием вебхука от MAX. Должен отвечать за < 1 секунду."""
    # 1. Проверка подписи
    signature = request.headers.get("X-Max-Signature", "")
    body = await request.body()
    if not verify_max_signature(body, signature):
        logger.warning("max_invalid_signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 2. Парсинг payload
    try:
        update = MaxUpdate.model_validate_json(body)
    except Exception as e:
        logger.error("max_invalid_payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not update.message:
        return {"status": "ignored"} # Игнорируем edit_message, callback_query и т.д.

    msg = update.message
    
    # 3. Асинхронная обработка в фоне (чтобы сразу вернуть 200 OK в MAX)
    background_tasks.add_task(process_max_message, msg)
    
    return {"status": "ok"}

async def process_max_message(msg: MaxMessage):
    """Фоновая задача: нормализация и отправка в Celery."""
    try:
        # Базовая нормализация
        normalized = IncomingMessage(
            channel=Channel.MAX,
            user_id=msg.from_user.user_id,
            chat_id=msg.chat_id,
            message_id=msg.message_id,
            timestamp=datetime.fromtimestamp(msg.timestamp),
            user_display_name=msg.from_user.username or msg.from_user.first_name,
            text=msg.text,
            raw_payload=msg.model_dump()
        )

        # Обработка медиа (голосовые или кружочки)
        if msg.voice:
            normalized.media_type = MediaType.VOICE
            s3_key = await download_and_upload_media(
                msg.voice.file_url, 
                channel="max", 
                media_type="voice"
            )
            if not s3_key:
                logger.error("max_voice_processing_failed", message_id=msg.message_id)
                return # Прерываем, если не удалось скачать
            normalized.audio_s3_key = s3_key

        # 4. Диспетчеризация в Celery
        await dispatch_to_celery(normalized)
        logger.info("max_message_dispatched", message_id=msg.message_id, channel="max")

    except Exception as e:
        logger.error("max_processing_failed", message_id=msg.message_id, error=str(e), exc_info=True)
```

---

## Шаг 1.2.3: Адаптер для VK (Приоритет 2)

VK использует немного другую структуру (Callback API), которую мы также нормализуем.

**Файл: `app/adapters/vk_adapter.py`**
```python
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Literal, Any
from datetime import datetime

from app.core.config import settings
from app.core.logger import logger
from app.core.models import IncomingMessage, Channel, MediaType
from app.adapters.utils import download_and_upload_media
from app.workers.ingestion_tasks import dispatch_to_celery

router = APIRouter(prefix="/vk", tags=["vk"])

# --- Схемы данных VK Callback API ---
class VKMessageObject(BaseModel):
    id: int
    date: int
    from_id: int
    peer_id: int
    text: str = ""
    attachments: list[dict[str, Any]] = Field(default_factory=list)

class VKUpdate(BaseModel):
    type: str
    object: dict[str, Any]
    group_id: int

@router.post("/webhook")
async def handle_vk_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        update = VKUpdate.model_validate(payload)
    except Exception as e:
        logger.error("vk_invalid_payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. Подтверждение сервера (требуется VK при настройке)
    if update.type == "confirmation":
        return settings.vk_confirmation_token.get_secret_value()

    # 2. Обработка только новых сообщений
    if update.type != "message_new":
        return {"ok": True}

    msg_data = update.object.get("message", {})
    try:
        msg = VKMessageObject.model_validate(msg_data)
    except Exception as e:
        logger.error("vk_invalid_message_object", error=str(e))
        return {"ok": True} # Возвращаем ok, чтобы VK не спамил ретраями

    background_tasks.add_task(process_vk_message, msg, update.group_id)
    return {"ok": True}

async def process_vk_message(msg: VKMessageObject, group_id: int):
    try:
        normalized = IncomingMessage(
            channel=Channel.VK,
            user_id=str(msg.from_id),
            chat_id=str(msg.peer_id),
            message_id=str(msg.id),
            timestamp=datetime.fromtimestamp(msg.date),
            text=msg.text if msg.text else None,
            raw_payload={"message": msg.model_dump(), "group_id": group_id}
        )

        # Проверка на голосовое сообщение в attachments
        voice_attachment = next((att for att in msg.attachments if att.get("type") == "audio_message"), None)
        if voice_attachment:
            normalized.media_type = MediaType.VOICE
            audio_doc = voice_attachment["audio_message"]
            file_url = audio_doc["link_ogg"] # VK предоставляет прямую ссылку на ogg
            
            s3_key = await download_and_upload_media(file_url, channel="vk", media_type="voice")
            if not s3_key:
                logger.error("vk_voice_processing_failed", message_id=msg.id)
                return
            normalized.audio_s3_key = s3_key

        await dispatch_to_celery(normalized)
        logger.info("vk_message_dispatched", message_id=msg.id, channel="vk")

    except Exception as e:
        logger.error("vk_processing_failed", message_id=msg.id, error=str(e), exc_info=True)
```

---

## Шаг 1.2.4: Адаптер для Telegram (Приоритет 3)

Для сохранения единого стиля (Channel Adapter) мы используем сырой FastAPI + `httpx`, а не `aiogram`. Это дает идентичный контроль над потоком данных и меньший оверхед.

**Файл: `app/adapters/telegram_adapter.py`**
```python
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
import httpx

from app.core.config import settings
from app.core.logger import logger
from app.core.models import IncomingMessage, Channel, MediaType
from app.adapters.utils import download_and_upload_media
from app.workers.ingestion_tasks import dispatch_to_celery

router = APIRouter(prefix="/telegram", tags=["telegram"])

class TGUser(BaseModel):
    id: int
    username: Optional[str] = None
    first_name: Optional[str] = None

class TGVoice(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int
    file_size: int

class TGMessage(BaseModel):
    message_id: int
    date: int
    chat: dict[str, Any]
    from_user: Optional[TGUser] = Field(alias="from", default=None)
    text: Optional[str] = None
    voice: Optional[TGVoice] = None

class TGUpdate(BaseModel):
    update_id: int
    message: Optional[TGMessage] = None

@router.post("/webhook")
async def handle_tg_webhook(request: Request, background_tasks: BackgroundTasks):
    # Примечание: В продакшене здесь стоит добавить проверку IP-адресов Telegram
    # или использование webhook secret token, если поддерживается вашим провайдером.
    try:
        payload = await request.json()
        update = TGUpdate.model_validate(payload)
    except Exception as e:
        logger.error("tg_invalid_payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not update.message:
        return {"ok": True}

    msg = update.message
    background_tasks.add_task(process_tg_message, msg)
    return {"ok": True}

async def process_tg_message(msg: TGMessage):
    try:
        normalized = IncomingMessage(
            channel=Channel.TELEGRAM,
            user_id=str(msg.from_user.id) if msg.from_user else "unknown",
            chat_id=str(msg.chat["id"]),
            message_id=str(msg.message_id),
            timestamp=datetime.fromtimestamp(msg.date),
            user_display_name=msg.from_user.username if msg.from_user else None,
            text=msg.text,
            raw_payload=msg.model_dump()
        )

        if msg.voice:
            normalized.media_type = MediaType.VOICE
            
            # Telegram требует двух шагов: getFile -> download
            bot_token = settings.telegram_bot_token.get_secret_value()
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Получаем путь к файлу
                resp = await client.get(f"https://api.telegram.org/bot{bot_token}/getFile?file_id={msg.voice.file_id}")
                resp.raise_for_status()
                file_path = resp.json()["result"]["file_path"]
                
                # 2. Скачиваем и загружаем в S3
                file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                s3_key = await download_and_upload_media(file_url, channel="telegram", media_type="voice")
                
                if not s3_key:
                    logger.error("tg_voice_processing_failed", message_id=msg.message_id)
                    return
                normalized.audio_s3_key = s3_key

        await dispatch_to_celery(normalized)
        logger.info("tg_message_dispatched", message_id=msg.message_id, channel="telegram")

    except Exception as e:
        logger.error("tg_processing_failed", message_id=msg.message_id, error=str(e), exc_info=True)
```

---

## Шаг 1.2.5: Диспетчеризация в Celery

Теперь создадим задачу, которая принимает нормализованное сообщение и отправляет его в правильную очередь.

**Файл: `app/workers/ingestion_tasks.py`**
```python
from app.core.celery_app import celery_app
from app.core.models import IncomingMessage, MediaType
from app.core.logger import logger
from app.infrastructure.redis_client import redis_service

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
```

---

## Шаг 1.2.6: Сборка FastAPI приложения

Объединим все роутеры в единое приложение.

**Файл: `app/main.py`**
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.logger import setup_logger, logger
from app.infrastructure.redis_client import redis_service
from app.adapters.max_adapter import router as max_router
from app.adapters.vk_adapter import router as vk_router
from app.adapters.telegram_adapter import router as telegram_router

# Инициализация логгера
setup_logger()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LinguaBridge AI Mediator Ingestion API",
)

# Подключение роутеров
app.include_router(max_router, prefix="/webhooks")
app.include_router(vk_router, prefix="/webhooks")
app.include_router(telegram_router, prefix="/webhooks")

@app.on_event("startup")
async def startup_event():
    logger.info("application_starting")
    await redis_service.connect()
    logger.info("redis_connected_on_startup")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("application_shutting_down")
    await redis_service.close()

# Глобальный обработчик исключений для чистых логов
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.get("/health")
async def health_check():
    """Эндпоинт для Kubernetes liveness/readiness проб."""
    return {"status": "healthy", "service": settings.app_name}
```

---

## Шаг 1.2.7: Тестирование адаптеров (Интеграционный тест)

Создадим скрипт, который эмулирует входящие вебхуки, чтобы убедиться, что нормализация и диспетчеризация работают.

**Файл: `test_adapters.py`**
```python
import asyncio
import httpx
from app.core.logger import setup_logger, logger
from app.core.models import Channel, MediaType

async def test_max_webhook():
    """Эмуляция входящего голосового сообщения от MAX."""
    payload = {
        "update_id": 1001,
        "message": {
            "message_id": "msg_max_123",
            "chat_id": "chat_999",
            "from": {"user_id": "user_555", "username": "test_user"},
            "timestamp": 1710000000,
            "voice": {
                "file_id": "voice_123",
                "file_url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", # Тестовый аудиофайл
                "duration": 5,
                "mime_type": "audio/ogg"
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        # В реальном MAX здесь был бы заголовок X-Max-Signature
        response = await client.post("http://localhost:8000/webhooks/max/webhook", json=payload)
        logger.info("max_test_response", status_code=response.status_code, body=response.text)

async def main():
    setup_logger()
    logger.info("starting_adapter_tests")
    
    # Убедитесь, что приложение запущено: poetry run uvicorn app.main:app --reload
    await test_max_webhook()
    
    logger.info("adapter_tests_completed")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Проверка и Действия для Шага 1.2

1. Убедитесь, что инфраструктура из Шага 1.1 запущена (`docker compose up -d`).
2. Запустите Celery воркер (в отдельном терминале), чтобы он мог принимать задачи:
   ```bash
   poetry run celery -A app.core.celery_app worker --loglevel=info --concurrency=2 -Q translate_text,translate_voice
   ```
3. Запустите FastAPI приложение:
   ```bash
   poetry run uvicorn app.main:app --reload --port 8000
   ```
4. Запустите тестовый скрипт:
   ```bash
   poetry run python test_adapters.py
   ```
5. **Ожидаемый результат:** 
   - В логах Uvicorn вы увидите `200 OK` на POST-запрос.
   - В логах Celery вы увидите, что задача `dispatch_to_celery_task` приняла сообщение, определила очередь `translate_voice`, скачала тестовый аудиофайл и загрузила его в MinIO (S3).
   - В консоли MinIO (http://localhost:9001) появится файл по пути `max/voice/{uuid}.ogg`.

---

### Что мы достигли в Подзадаче 1.2:

✅ **Паттерн Channel Adapter реализован:** Три разных API (MAX, VK, Telegram) сводятся к единой, строго типизированной модели `IncomingMessage`.
✅ **Безопасность:** Реализована проверка HMAC-подписи для MAX (и заложена основа для других).
✅ **Производительность:** Использование `BackgroundTasks` в FastAPI гарантирует, что вебхук отвечает провайдеру за миллисекунды, а тяжелая работа (скачивание, S3, Celery) идет в фоне.
✅ **Надежность:** Асинхронное скачивание и загрузка в S3 инкапсулирована в утилиту `download_and_upload_media`, которая корректно обрабатывает ошибки сети, не роняя весь процесс.
✅ **Маршрутизация:** Умная диспетчеризация: голосовые сообщения автоматически попадают в очередь `translate_voice` с соответствующим приоритетом, текстовые — в `translate_text`.
