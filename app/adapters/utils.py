import uuid

import httpx

from app.core.logger import logger
from app.infrastructure.s3 import s3_service


async def download_and_upload_media(file_url: str, channel: str, media_type: str) -> str | None:
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
