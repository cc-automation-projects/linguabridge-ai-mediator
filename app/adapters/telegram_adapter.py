from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.adapters.utils import download_and_upload_media
from app.core.config import settings
from app.core.logger import logger
from app.core.models import Channel, IncomingMessage, MediaType
from app.workers.ingestion_tasks import dispatch_to_celery

router = APIRouter(prefix="/telegram", tags=["telegram"])


class TGUser(BaseModel):
    id: int
    username: str | None = None
    first_name: str | None = None


class TGVoice(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int
    file_size: int


class TGMessage(BaseModel):
    message_id: int
    date: int
    chat: dict[str, Any]
    from_user: TGUser | None = Field(alias="from", default=None)
    text: str | None = None
    voice: TGVoice | None = None


class TGUpdate(BaseModel):
    update_id: int
    message: TGMessage | None = None


@router.post("/webhook")
async def handle_tg_webhook(request: Request, background_tasks: BackgroundTasks):
    # Примечание: В продакшене здесь стоит добавить проверку IP-адресов Telegram
    # или использование webhook secret token, если поддерживается вашим провайдером.
    try:
        payload = await request.json()
        update = TGUpdate.model_validate(payload)
    except Exception as e:
        logger.error("tg_invalid_payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

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
