from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.adapters.utils import download_and_upload_media
from app.core.config import settings
from app.core.logger import logger
from app.core.models import Channel, IncomingMessage, MediaType
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
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

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
