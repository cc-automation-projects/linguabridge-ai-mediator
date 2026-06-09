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
