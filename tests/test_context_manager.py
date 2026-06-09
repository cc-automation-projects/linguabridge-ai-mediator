import asyncio

import pytest

from app.core.models import Channel, ConversationTurn
from app.infrastructure.context_manager import context_manager
from app.infrastructure.redis_client import redis_service


@pytest.fixture(autouse=True)
async def cleanup_redis():
    await redis_service.connect()
    yield
    keys = await redis_service.client.keys("linguabridge:ctx:*")
    if keys:
        await redis_service.client.delete(*keys)
    await redis_service.close()


@pytest.mark.asyncio
async def test_context_accumulation_and_truncation():
    user_id = "test_user_1"
    channel = Channel.MAX

    for i in range(12):
        turn = ConversationTurn(
            role="client",
            original_lang="uz",
            original_text=f"Сообщение {i}",
            translated_text=f"Message {i}"
        )
        await context_manager.add_turn(user_id, channel, turn)

    ctx = await context_manager.get_context(user_id, channel)

    assert len(ctx.turns) == 10
    assert ctx.turns[0].original_text == "Сообщение 2"
    assert ctx.turns[-1].original_text == "Сообщение 11"


@pytest.mark.asyncio
async def test_context_ttl_expiration():
    user_id = "test_user_2"
    channel = Channel.VK

    turn = ConversationTurn(role="client", original_lang="ru", original_text="Привет", translated_text="Hi")

    original_ttl = context_manager.ttl_seconds
    context_manager.ttl_seconds = 1

    await context_manager.add_turn(user_id, channel, turn)

    ctx = await context_manager.get_context(user_id, channel)
    assert len(ctx.turns) == 1

    await asyncio.sleep(1.5)

    ctx_expired = await context_manager.get_context(user_id, channel)
    assert len(ctx_expired.turns) == 0
    assert ctx_expired.user_id == user_id

    context_manager.ttl_seconds = original_ttl


@pytest.mark.asyncio
async def test_channel_isolation():
    user_id = "test_user_3"

    turn_max = ConversationTurn(role="client", original_lang="uz", original_text="MAX text", translated_text="MAX trans")
    turn_vk = ConversationTurn(role="client", original_lang="ru", original_text="VK text", translated_text="VK trans")

    await context_manager.add_turn(user_id, Channel.MAX, turn_max)
    await context_manager.add_turn(user_id, Channel.VK, turn_vk)

    ctx_max = await context_manager.get_context(user_id, Channel.MAX)
    ctx_vk = await context_manager.get_context(user_id, Channel.VK)

    assert len(ctx_max.turns) == 1
    assert ctx_max.turns[0].original_text == "MAX text"

    assert len(ctx_vk.turns) == 1
    assert ctx_vk.turns[0].original_text == "VK text"
