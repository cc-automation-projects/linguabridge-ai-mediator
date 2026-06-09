
import redis.asyncio as redis

from app.core.config import settings


class RedisClient:
    """Обертка над redis.asyncio с оптимальными настройками."""

    def __init__(self) -> None:
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None

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
