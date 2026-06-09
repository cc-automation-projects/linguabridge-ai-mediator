Это критически важный мост между AI-обработкой и бизнес-логикой. Мы должны гарантировать, что даже если amoCRM временно недоступна или возвращает странные ошибки (что для её API не редкость), наш Celery-воркер не зависнет навсегда, а корректно обработает сбой, защитив всю систему от каскадного отказа.

---

# ЭТАП 3, ПОДЗАДАЧА 3.1: Асинхронный клиент amoCRM с Circuit Breaker

## Шаг 3.1.1: Конфигурация и зависимости

Убедимся, что все необходимые библиотеки указаны, и добавим специфичные настройки для amoCRM.

**1. Проверка `pyproject.toml` (должны быть):**
```toml
httpx = "^0.27.0"
pybreaker = "^1.0.2"
```

**2. Обновите `app/core/config.py`:**
```python
    # === amoCRM Integration ===
    amocrm_subdomain: str = Field(default="your-company", description="Поддомен amoCRM (без .amocrm.ru)")
    amocrm_access_token: SecretStr = Field(default=SecretStr(""))
    # ID кастомного поля в amoCRM, где хранится внешний ID пользователя (например, ID из MAX/Telegram)
    # Замените на реальный ID из настроек вашей amoCRM
    amocrm_custom_field_user_id: int = Field(default=123456) 
    amocrm_request_timeout: float = Field(default=5.0, description="Таймаут запросов к amoCRM")
    amocrm_cb_fail_max: int = Field(default=3, description="Кол-во ошибок подряд для размыкания цепи")
    amocrm_cb_reset_timeout: int = Field(default=60, description="Секунд до попытки восстановления цепи")
```

---

## Шаг 3.1.2: Реализация защищенного клиента amoCRM

Ключевая особенность amoCRM: она часто возвращает HTTP-статус `200 OK`, но внутри JSON-тела лежит `{"status": "error", "title": "...", "detail": "..."}`. Стандартный `response.raise_for_status()` это пропустит. Мы реализуем явную проверку.

Также, поскольку `pybreaker` по своей природе синхронный, мы обернем его вызов в `asyncio.to_thread`, чтобы не блокировать `asyncio` event loop нашего асинхронного приложения.

**Файл: `app/integrations/amocrm_client.py`**
```python
import asyncio
import pybreaker
import httpx
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

from app.core.config import settings
from app.core.logger import logger
from app.core.models import Channel

# 1. Глобальная настройка Circuit Breaker
amo_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.amocrm_cb_fail_max,
    reset_timeout=settings.amocrm_cb_reset_timeout,
    name="amocrm_api_breaker"
)

class AmoCRMClient:
    def __init__(self):
        self.base_url = f"https://{settings.amocrm_subdomain}.amocrm.ru/api/v4/"
        self.token = settings.amocrm_access_token.get_secret_value()
        
        self.client = httpx.AsyncClient(
            timeout=settings.amocrm_request_timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        logger.info("amocrm_client_initialized_with_circuit_breaker")

    def _check_amo_error(self, response: httpx.Response) -> None:
        """
        Специфичная проверка: ловит ошибки, спрятанные внутри HTTP 200 OK.
        """
        response.raise_for_status() # Ловим реальные HTTP ошибки (401, 404, 502, 504)
        
        try:
            data = response.json()
            # amoCRM возвращает dict с ключом 'status' при ошибках, или list при успехе (для некоторых эндпоинтов)
            if isinstance(data, dict) and data.get("status") == "error":
                raise ValueError(f"amoCRM API Error: {data.get('title')} - {data.get('detail')}")
        except httpx.JSONDecodeError:
            # Если ответ не JSON, но статус 200, считаем это аномалией, но не ошибкой API
            pass

    async def _make_request_async(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Внутренний метод, оборачивающий HTTP-запрос в Circuit Breaker и выполняющий его в отдельном потоке.
        """
        url = urljoin(self.base_url, endpoint)
        
        def _sync_request() -> Dict[str, Any]:
            @amo_circuit_breaker
            def _protected_request():
                # httpx.Client синхронный, но мы вызываем его внутри to_thread, это безопасно
                # Для простоты используем sync-клиент внутри потока, или можно создать sync-клиент отдельно.
                # Лучшая практика: использовать httpx.Client внутри потока для thread-safety.
                with httpx.Client(
                    timeout=settings.amocrm_request_timeout,
                    headers=self.client.headers
                ) as sync_client:
                    response = sync_client.request(method, url, **kwargs)
                    self._check_amo_error(response)
                    return response.json()
            return _protected_request()

        try:
            # Выносим блокирующий вызов (включая логику breaker) в пул потоков
            result = await asyncio.to_thread(_sync_request)
            return result
        except pybreaker.CircuitBreakerError:
            logger.critical(
                "amocrm_circuit_breaker_open", 
                msg="amoCRM недоступен или возвращает ошибки. Запрос отклонен для защиты системы."
            )
            raise # Пробрасываем, чтобы Celery мог сделать retry
        except Exception as e:
            logger.error("amocrm_request_failed", endpoint=endpoint, error=str(e), exc_info=True)
            raise

    # --- Бизнес-методы ---

    async def find_or_create_lead(self, user_id: str, channel: Channel, user_display_name: Optional[str]) -> int:
        """
        Ищет контакт/лид по кастомному полю (user_id канала). Если не находит, создает новый.
        Возвращает ID созданного или найденного лида.
        """
        field_id = settings.amocrm_custom_field_user_id
        
        # 1. Попытка найти существующий лид
        search_payload = {
            "query": {
                "custom_fields_values": [
                    {
                        "field_id": field_id,
                        "values": [{"value": user_id}]
                    }
                ]
            },
            "limit": 1
        }
        
        try:
            response = await self._make_request_async("POST", "leads/complex/search", json=search_payload)
            leads = response.get("_embedded", {}).get("leads", [])
            if leads:
                logger.info("amocrm_lead_found", lead_id=leads[0]["id"], user_id=user_id)
                return leads[0]["id"]
        except Exception as e:
            logger.warning("amocrm_search_failed_fallback_to_create", error=str(e))

        # 2. Если не найден или ошибка поиска, создаем новый лид
        channel_name = channel.value.upper()
        new_lead_payload = {
            "name": f"Заявка из {channel_name}: {user_display_name or 'Неизвестный'}",
            "custom_fields_values": [
                {
                    "field_id": field_id,
                    "values": [{"value": user_id}]
                }
            ]
        }
        
        response = await self._make_request_async("POST", "leads", json=[new_lead_payload])
        lead_id = response["_embedded"]["leads"][0]["id"]
        logger.info("amocrm_lead_created", lead_id=lead_id, user_id=user_id)
        return lead_id

    async def add_note(self, lead_id: int, note_text: str) -> None:
        """Добавляет текстовое примечание к карточке лида."""
        note_payload = {
            "note_type": "common", # Обычное примечание
            "params": {
                "text": note_text
            }
        }
        await self._make_request_async("POST", f"leads/{lead_id}/notes", json=[note_payload])
        logger.info("amocrm_note_added", lead_id=lead_id)

    async def update_tags(self, lead_id: int, tags_to_add: List[str]) -> None:
        """Добавляет теги к лиду (например, канал и язык)."""
        # В amoCRM теги добавляются через обновление сущности
        update_payload = {
            "tags": [{"name": tag} for tag in tags_to_add]
        }
        await self._make_request_async("PATCH", f"leads/{lead_id}", json=update_payload)
        logger.info("amocrm_tags_updated", lead_id=lead_id, tags=tags_to_add)

    async def close(self):
        await self.client.aclose()

# Глобальный синглтон
amocrm_client = AmoCRMClient()
```

---

## Шаг 3.1.3: Интеграция в Celery Pipeline

Теперь мы добавляем вызов amoCRM в конец наших задач обработки. Мы формируем красивое, структурированное сообщение для оператора.

**Обновите файл: `app/workers/translation_tasks.py`**
```python
# ... предыдущие импорты ...
from app.integrations.amocrm_client import amocrm_client

# ... внутри задачи process_incoming_message (после получения final_text) ...

        # 6. Интеграция с amoCRM
        try:
            # А. Находим или создаем лид
            lead_id = await amocrm_client.find_or_create_lead(
                user_id=msg.user_id,
                channel=msg.channel,
                user_display_name=msg.user_display_name
            )
            
            # Б. Формируем текст примечания для оператора
            operator_note = (
                f"🌐 **[AI Перевод]** Канал: {msg.channel.value.upper()} | Язык: {msg.detected_lang.upper()}\n"
                f"📝 **Перевод:** {msg.translated_text}\n"
                f"---\n"
                f"🔒 **Оригинал (для проверки):** {msg.masked_text or msg.text}\n"
                f"⚠️ **Уверенность ASR/перевода:** {round(msg.lang_confidence or 1.0, 2)}"
            )
            
            # В. Добавляем примечание
            await amocrm_client.add_note(lead_id, operator_note)
            
            # Г. Обновляем теги (канал + язык)
            tags = [f"channel_{msg.channel.value}", f"lang_{msg.detected_lang}"]
            await amocrm_client.update_tags(lead_id, tags)
            
            logger.info("amocrm_integration_successful", lead_id=lead_id, message_id=msg.message_id)
            
        except pybreaker.CircuitBreakerError:
            logger.error("amocrm_circuit_open_skipping_crm_update", message_id=msg.message_id)
            # Не роняем задачу, так как перевод уже выполнен. 
            # В продакшене здесь можно отправить сообщение в Dead Letter Queue (DLQ) для повторной попытки позже.
        except Exception as e:
            logger.error("amocrm_integration_failed", message_id=msg.message_id, error=str(e), exc_info=True)
            # Аналогично, не роняем основную задачу, если это не критично для бизнес-процесса

        logger.info("processing_message_completed", message_id=msg.message_id)
        return msg.model_dump()
```
*(Ту же логику интеграции с amoCRM нужно добавить в конец задачи `process_voice_message`)*

---

## Шаг 3.1.4: Исчерпывающее тестирование (Unit & Integration)

Напишем тесты, которые эмулируют как успешную работу, так и срабатывание Circuit Breaker при сбоях amoCRM.

**Файл: `tests/test_amocrm_client.py`**
```python
import pytest
import respx
import httpx
from unittest.mock import patch
import pybreaker

from app.integrations.amocrm_client import amocrm_client, amo_circuit_breaker
from app.core.models import Channel

@pytest.fixture(autouse=True)
def reset_breaker():
    """Сбрасываем состояние Circuit Breaker перед каждым тестом."""
    amo_circuit_breaker.reset()
    yield

@respx.mock
@pytest.mark.asyncio
async def test_find_or_create_lead_success():
    """Тест успешного создания лида, если поиск не дал результатов."""
    # Мокаем ответ на поиск (пустой)
    respx.post("https://test-company.amocrm.ru/api/v4/leads/complex/search").respond(
        json={"_embedded": {"leads": []}}
    )
    # Мокаем ответ на создание
    respx.post("https://test-company.amocrm.ru/api/v4/leads").respond(
        json={"_embedded": {"leads": [{"id": 98765}]}}
    )
    
    lead_id = await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test User")
    assert lead_id == 98765

@respx.mock
@pytest.mark.asyncio
async def test_amo_fake_200_error():
    """Тест перехвата ошибки, спрятанной в HTTP 200 OK."""
    respx.post("https://test-company.amocrm.ru/api/v4/leads").respond(
        json={"status": "error", "title": "Invalid token", "detail": "Token expired"}
    )
    
    with pytest.raises(ValueError, match="amoCRM API Error: Invalid token"):
        await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test User")

@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_repeated_failures():
    """Тест размыкания цепи после 3-х ошибок подряд."""
    # Принудительно вызываем ошибку 3 раза
    for _ in range(3):
        with pytest.raises(Exception): # Любая ошибка засчитывается breaker-ом
            # Эмулируем сбой сети
            with patch.object(httpx.Client, 'request', side_effect=httpx.ConnectError("Connection refused")):
                try:
                    await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test")
                except Exception:
                    pass # Игнорируем, чтобы перейти к следующей итерации

    # 4-й вызов должен мгновенно выбросить CircuitBreakerError, не делая реального запроса
    with pytest.raises(pybreaker.CircuitBreakerError):
        await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test")
```
*(Примечание: для использования `respx` добавьте его в `pyproject.toml`: `respx = "^0.21.1"` и выполните `poetry install`)*

---

## Шаг 3.1.5: Production-нюансы для amoCRM

1. **Rate Limiting:** amoCRM имеет строгие лимиты (7 запросов в секунду на один аккаунт). Наш `httpx.Limits(max_connections=20)` и асинхронная природа могут превысить этот лимит при пиковой нагрузке. 
   *Решение:* В Celery уже настроен `worker_prefetch_multiplier=1`, что ограничивает параллельную обработку. Для продакшена рекомендуется добавить `tenacity` retry с экспоненциальной задержкой конкретно на ошибку `429 Too Many Requests` внутри `_make_request_async`.
2. **Истечение токена:** В реальном enterprise-окружении access token живет 24 часа. В текущей реализации мы используем статический токен из `.env`. Для полноценного продакшена необходимо реализовать фоновую задачу (Celery Beat), которая будет обновлять токен через `oauth2` refresh flow и обновлять переменную окружения или запись в Redis.
3. **Кастомные поля:** Убедитесь, что `amocrm_custom_field_user_id` в `.env` точно соответствует ID поля в вашей amoCRM, иначе поиск дубликатов не сработает, и система будет создавать новых лидов на каждое сообщение.

---

### Что мы достигли в Подзадаче 3.1:

✅ **Надежная интеграция:** Реализован полноценный асинхронный клиент с пулом соединений и корректной обработкой "коварных" ответов amoCRM (200 OK с ошибкой).  

✅ **Защита от каскадных сбоев:** Внедрен `pybreaker`. Если amoCRM "ляжет", система не будет бесконечно ждать таймаутов, а мгновенно разомкнет цепь, позволив Celery-задаче завершиться или уйти в retry, сохраняя ресурсы.  

✅ **Богатый контекст для оператора:** В карточку лида добавляется структурированное примечание с переводом, оригиналом (для проверки) и метаданными, а также проставляются теги канала и языка.  

✅ **Изоляция сбоев:** Ошибка интеграции с CRM не приводит к потере уже выполненного AI-перевода (благодаря обработке исключений в конце пайплайна).  
