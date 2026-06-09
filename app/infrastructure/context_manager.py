from app.core.logger import logger
from app.core.models import Channel, ConversationContext, ConversationTurn
from app.infrastructure.redis_client import redis_service


class ContextManager:
    def __init__(self, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        self.key_prefix = "linguabridge:ctx"
        logger.info("context_manager_initialized", ttl_hours=ttl_hours)

    def _get_redis_key(self, user_id: str, channel: Channel) -> str:
        return f"{self.key_prefix}:{channel.value}:{user_id}"

    async def get_context(self, user_id: str, channel: Channel) -> ConversationContext:
        key = self._get_redis_key(user_id, channel)
        data = await redis_service.get_json(key)

        if data:
            try:
                return ConversationContext.model_validate(data)
            except Exception as e:
                logger.warning("context_validation_failed", user_id=user_id, channel=channel.value, error=str(e))
                return ConversationContext(user_id=user_id, channel=channel)

        return ConversationContext(user_id=user_id, channel=channel)

    async def add_turn(self, user_id: str, channel: Channel, turn: ConversationTurn) -> None:
        key = self._get_redis_key(user_id, channel)

        ctx = await self.get_context(user_id, channel)
        ctx.add_turn(turn)

        await redis_service.set_json(key, ctx.model_dump(), ttl=self.ttl_seconds)

        logger.debug(
            "context_updated",
            user_id=user_id,
            channel=channel.value,
            turns_count=len(ctx.turns)
        )

    async def clear_context(self, user_id: str, channel: Channel) -> None:
        key = self._get_redis_key(user_id, channel)
        await redis_service.delete(key)
        logger.info("context_cleared", user_id=user_id, channel=channel.value)


context_manager = ContextManager(ttl_hours=24)
