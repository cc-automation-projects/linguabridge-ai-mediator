from typing import Any

from app.core.logger import logger
from app.core.redis import redis_client


class RedisService:
    @property
    def client(self):
        return redis_client.client

    async def connect(self) -> None:
        await redis_client.connect()

    async def close(self) -> None:
        await redis_client.close()

    async def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            data = await redis_client.client.get(key)
            if data is not None:
                import json
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("redis_get_json_failed", key=key, error=str(e), exc_info=True)
            return None

    async def set_json(self, key: str, value: dict, ttl: int | None = None) -> None:
        import json
        try:
            data = json.dumps(value, default=str)
            await redis_client.client.set(key, data, ex=ttl)
        except Exception as e:
            logger.error("redis_set_json_failed", key=key, error=str(e), exc_info=True)
            raise

    async def delete(self, key: str) -> None:
        try:
            await redis_client.client.delete(key)
        except Exception as e:
            logger.error("redis_delete_failed", key=key, error=str(e), exc_info=True)


redis_service = RedisService()
