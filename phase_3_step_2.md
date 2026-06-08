Это критически важный компонент для качества перевода. Без контекста модель NLLB будет переводить местоимения ("он", "она", "это") и разговорные фразы изолированно, что приведет к "рваному" и неестественному тексту. Мы реализуем атомарное, быстрое и надежное хранение последних 10 реплик с автоматическим истечением срока жизни (TTL).

---

# ЭТАП 3, ПОДЗАДАЧА 3.2: Управление контекстом диалога в Redis

## Шаг 3.2.1: Уточнение Pydantic-моделей

Убедимся, что наши модели из Шага 1.1 идеально подходят для сериализации в JSON и хранения в Redis. Используем `datetime.now(timezone.utc)` вместо устаревшего `utcnow()` (Python 3.12).

**Обновите `app/core/models.py` (добавьте/проверьте):**
```python
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, List
from pydantic import BaseModel, Field

class Channel(str, Enum):
    MAX = "max"
    VK = "vk"
    TELEGRAM = "telegram"

class ConversationTurn(BaseModel):
    """Одна реплика в контексте диалога."""
    role: str  # "client" или "operator"
    original_lang: str
    original_text: str
    translated_text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConversationContext(BaseModel):
    """Контекст диалога для сохранения в Redis."""
    user_id: str
    channel: Channel
    turns: List[ConversationTurn] = Field(default_factory=list, max_length=10)
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_turn(self, turn: ConversationTurn) -> None:
        """Добавляет реплику, сохраняя максимум 10 последних (FIFO)."""
        self.turns.append(turn)
        if len(self.turns) > 10:
            self.turns = self.turns[-10:] # Оставляем только последние 10
        self.last_activity_at = datetime.now(timezone.utc)
```

---

## Шаг 3.2.2: Реализация сервиса управления контекстом

Мы создадим специализированный сервис, который инкапсулирует логику чтения, обновления и очистки контекста.

**Файл: `app/infrastructure/context_manager.py`**
```python
import logging
from datetime import datetime, timezone, timedelta
from app.core.models import ConversationContext, ConversationTurn, Channel
from app.infrastructure.redis_client import redis_service
from app.core.logger import logger

class ContextManager:
    def __init__(self, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        self.key_prefix = "linguabridge:ctx"
        logger.info("context_manager_initialized", ttl_hours=ttl_hours)

    def _get_redis_key(self, user_id: str, channel: Channel) -> str:
        """Генерирует уникальный ключ для Redis."""
        return f"{self.key_prefix}:{channel.value}:{user_id}"

    async def get_context(self, user_id: str, channel: Channel) -> ConversationContext:
        """
        Получает контекст диалога. Если его нет, возвращает пустой, но инициализированный объект.
        """
        key = self._get_redis_key(user_id, channel)
        data = await redis_service.get_json(key)
        
        if data:
            try:
                return ConversationContext.model_validate(data)
            except Exception as e:
                logger.warning("context_validation_failed", user_id=user_id, channel=channel.value, error=str(e))
                # В случае повреждения данных возвращаем чистый контекст
                return ConversationContext(user_id=user_id, channel=channel)
        
        return ConversationContext(user_id=user_id, channel=channel)

    async def add_turn(self, user_id: str, channel: Channel, turn: ConversationTurn) -> None:
        """
        Добавляет новую реплику в контекст и обновляет TTL.
        """
        key = self._get_redis_key(user_id, channel)
        
        # 1. Получаем текущий контекст
        ctx = await self.get_context(user_id, channel)
        
        # 2. Добавляем новую реплику (метод add_turn сам обрежет список до 10 элементов)
        ctx.add_turn(turn)
        
        # 3. Сохраняем обратно в Redis с обновленным TTL
        # model_dump() сериализует datetime в ISO-формат, который Pydantic умеет читать обратно
        await redis_service.set_json(key, ctx.model_dump(), ttl=self.ttl_seconds)
        
        logger.debug(
            "context_updated", 
            user_id=user_id, 
            channel=channel.value, 
            turns_count=len(ctx.turns)
        )

    async def clear_context(self, user_id: str, channel: Channel) -> None:
        """Принудительная очистка контекста (например, по команде пользователя)."""
        key = self._get_redis_key(user_id, channel)
        await redis_service.delete(key)
        logger.info("context_cleared", user_id=user_id, channel=channel.value)

# Глобальный синглтон с TTL 24 часа (контекст "забывается" через сутки неактивности)
context_manager = ContextManager(ttl_hours=24)
```

---

## Шаг 3.2.3: Интеграция в Celery Pipeline

Теперь мы модифицируем задачи обработки сообщений, чтобы они читали контекст *перед* переводом (для будущей передачи в NLLB) и сохраняли новую реплику *после* успешного перевода.

**Обновите файл: `app/workers/translation_tasks.py`**
```python
import uuid
import asyncio
from datetime import datetime, timezone

from app.core.celery_app import celery_app
from app.core.models import IncomingMessage, ConversationTurn
from app.ml.whisper_asr import whisper_asr
from app.ml.language_detector import language_detector
from app.ml.pii_masker import pii_masker
from app.ml.nllb_translator import nllb_translator
from app.ml.terminology_override import terminology_override
from app.infrastructure.s3 import s3_service
from app.infrastructure.context_manager import context_manager
from app.integrations.amocrm_client import amocrm_client
from app.core.logger import logger
from app.core.context import set_trace_id, set_channel, set_user_id
import pybreaker

@celery_app.task(
    bind=True,
    name="app.workers.translation_tasks.process_incoming_message",
    queue="translate_text",
    acks_late=True,
    reject_on_worker_lost=True
)
def process_incoming_message(self, message_dict: dict) -> dict:
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    
    try:
        msg = IncomingMessage.model_validate(message_dict)
        set_channel(msg.channel.value)
        set_user_id(msg.user_id)
        
        logger.info("processing_message_started", message_id=msg.message_id, channel=msg.channel.value)

        if msg.text:
            # 1. Детекция языка и PII-маскирование
            detected_lang, confidence = language_detector.detect(msg.text)
            msg.detected_lang = detected_lang
            msg.lang_confidence = confidence
            
            target_lang_for_masking = detected_lang if confidence > 0.5 else "ru"
            msg.masked_text = pii_masker.mask(msg.text, lang=target_lang_for_masking)
            
            # 2. Получение контекста (для будущего использования в NLLB)
            ctx = asyncio.run(context_manager.get_context(msg.user_id, msg.channel))
            
            # 3. Перевод (упрощенно, без передачи контекста в NLLB пока, но структура готова)
            if detected_lang != "ru" and msg.masked_text:
                translated_text = asyncio.run(nllb_translator.translate(
                    text=msg.masked_text,
                    src_lang=detected_lang,
                    tgt_lang="ru"
                ))
            else:
                translated_text = msg.masked_text

            # 4. Terminology Override
            final_text = terminology_override.override(translated_text)
            msg.translated_text = final_text
            
            # 5. Сохранение реплики клиента в контекст
            client_turn = ConversationTurn(
                role="client",
                original_lang=detected_lang,
                original_text=msg.masked_text or msg.text,
                translated_text=final_text
            )
            asyncio.run(context_manager.add_turn(msg.user_id, msg.channel, client_turn))

            # 6. Интеграция с amoCRM
            try:
                lead_id = asyncio.run(amocrm_client.find_or_create_lead(
                    user_id=msg.user_id,
                    channel=msg.channel,
                    user_display_name=msg.user_display_name
                ))
                
                operator_note = (
                    f"🌐 **[AI Перевод]** Канал: {msg.channel.value.upper()} | Язык: {detected_lang.upper()}\n"
                    f"📝 **Перевод:** {final_text}\n"
                    f"---\n"
                    f"🔒 **Оригинал:** {msg.masked_text or msg.text}"
                )
                
                asyncio.run(amocrm_client.add_note(lead_id, operator_note))
                asyncio.run(amocrm_client.update_tags(lead_id, [f"channel_{msg.channel.value}", f"lang_{detected_lang}"]))
                
            except pybreaker.CircuitBreakerError:
                logger.error("amocrm_circuit_open_skipping", message_id=msg.message_id)
            except Exception as e:
                logger.error("amocrm_integration_failed", message_id=msg.message_id, error=str(e))

        logger.info("processing_message_completed", message_id=msg.message_id)
        return msg.model_dump()

    except Exception as e:
        logger.error("processing_message_failed", message_id=message_dict.get("message_id", "unknown"), error=str(e), exc_info=True)
        raise
```
*(Аналогичное обновление нужно внести в `process_voice_message`, добавляя `client_turn` после успешного распознавания и перевода).*

---

## Шаг 3.2.4: Исчерпывающее тестирование

Проверим, что контекст корректно накапливается, обрезается до 10 элементов и имеет правильный TTL.

**Файл: `tests/test_context_manager.py`**
```python
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from app.core.models import ConversationTurn, Channel
from app.infrastructure.context_manager import context_manager
from app.infrastructure.redis_client import redis_service

@pytest.fixture(autouse=True)
async def cleanup_redis():
    """Очистка тестовых ключей перед каждым тестом."""
    await redis_service.connect()
    yield
    # Очищаем все ключи с префиксом linguabridge:ctx
    keys = await redis_service.client.keys("linguabridge:ctx:*")
    if keys:
        await redis_service.client.delete(*keys)
    await redis_service.close()

@pytest.mark.asyncio
async def test_context_accumulation_and_truncation():
    """Проверка добавления реплик и обрезки до 10 элементов."""
    user_id = "test_user_1"
    channel = Channel.MAX
    
    # Добавляем 12 реплик
    for i in range(12):
        turn = ConversationTurn(
            role="client",
            original_lang="uz",
            original_text=f"Сообщение {i}",
            translated_text=f"Message {i}"
        )
        await context_manager.add_turn(user_id, channel, turn)
    
    # Получаем контекст
    ctx = await context_manager.get_context(user_id, channel)
    
    # Проверяем, что осталось ровно 10 последних реплик
    assert len(ctx.turns) == 10
    assert ctx.turns[0].original_text == "Сообщение 2" # Первая из оставшихся
    assert ctx.turns[-1].original_text == "Сообщение 11" # Последняя добавленная

@pytest.mark.asyncio
async def test_context_ttl_expiration():
    """Проверка истечения срока жизни контекста."""
    user_id = "test_user_2"
    channel = Channel.VK
    
    turn = ConversationTurn(role="client", original_lang="ru", original_text="Привет", translated_text="Hi")
    
    # Временно переопределяем TTL на 1 секунду для теста
    original_ttl = context_manager.ttl_seconds
    context_manager.ttl_seconds = 1
    
    await context_manager.add_turn(user_id, channel, turn)
    
    # Проверяем, что контекст есть
    ctx = await context_manager.get_context(user_id, channel)
    assert len(ctx.turns) == 1
    
    # Ждем истечения TTL
    await asyncio.sleep(1.5)
    
    # Проверяем, что контекст исчез и вернулся пустой
    ctx_expired = await context_manager.get_context(user_id, channel)
    assert len(ctx_expired.turns) == 0
    assert ctx_expired.user_id == user_id
    
    # Возвращаем TTL
    context_manager.ttl_seconds = original_ttl

@pytest.mark.asyncio
async def test_channel_isolation():
    """Проверка, что контексты разных каналов не смешиваются."""
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
```

**Запуск тестов:**
```bash
poetry run pytest tests/test_context_manager.py -v
```

---

## Шаг 3.2.5: Production-нюансы

1. **Атомарность (Race Conditions):** В текущей реализации используется паттерн "Read-Modify-Write". Если один пользователь отправит два сообщения *одновременно* (что редкость для чата, но возможно), может возникнуть состояние гонки. 
   *Решение для сверхвысоких нагрузок:* Использовать Redis Lists (`LPUSH` для добавления, `LTRIM 0 9` для обрезки) вместо JSON-объектов. Однако для чат-ботов текущий JSON-подход с Pydantic является стандартом де-факто из-за удобства типизации, а вероятность гонки для одного пользователя ничтожно мала.
2. **Управление памятью Redis:** TTL (24 часа) критически важен. Без него Redis превратится в свалку заброшенных диалогов. Убедитесь, что в `redis.conf` настроена политика вытеснения `maxmemory-policy allkeys-lru` на случай, если память все же закончится.
3. **Очистка по запросу:** Если клиент пишет "забыть всё" или "очистить историю", вы можете вызвать `context_manager.clear_context(user_id, channel)`, что мгновенно удалит данные из Redis.

---

### Что мы достигли в Подзадаче 3.2:

✅ **Строгая типизация и валидация:** Использование Pydantic V2 гарантирует, что в Redis попадут только корректные структуры данных, а при чтении они будут безопасно распарсены.  
✅ **Автоматическое управление размером:** Метод `add_turn` гарантирует, что контекст никогда не превысит 10 реплик, защищая от раздувания payload при передаче в LLM/NLLB.  
✅ **Экономия ресурсов:** TTL (24 часа) автоматически удаляет неактивные диалоги, предотвращая утечки памяти в Redis.  
✅ **Бесшовная интеграция:** Контекст теперь автоматически накапливается в конце каждого успешного пайплайна перевода, создавая готовую основу для улучшения качества перевода на следующем этапе (когда мы передадим этот контекст в NLLB).  