import hashlib
import hmac
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.adapters.utils import download_and_upload_media
from app.core.config import settings
from app.core.logger import logger
from app.core.models import Channel, IncomingMessage, MediaType
from app.workers.ingestion_tasks import dispatch_to_celery

router = APIRouter(prefix="/max", tags=["max"])

# --- Строгие схемы данных MAX ---
class MaxUser(BaseModel):
    user_id: str
    first_name: str | None = None
    username: str | None = None

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
    text: str | None = None
    voice: MaxVoice | None = None
    # Можно добавить photo, document и т.д. по аналогии

class MaxUpdate(BaseModel):
    update_id: int
    message: MaxMessage | None = None

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
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

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
