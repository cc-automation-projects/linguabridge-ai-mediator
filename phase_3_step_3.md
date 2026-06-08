Наша цель сейчас — превратить сырые данные, прошедшие через пайплайн AI, в **информационную панель**, которая позволит оператору за 1-2 секунды понять суть обращения, увидеть предупреждения (если они есть) и иметь доступ к оригиналу для разрешения спорных ситуаций. Мы вынесем логику форматирования в отдельный модуль для чистоты кода и легкости тестирования.

---

# ЭТАП 3, ПОДЗАДАЧА 3.3: Форматирование вывода для оператора и завершение Этапа 3

## Шаг 3.3.1: Модуль форматирования для amoCRM

Вместо конкатенации строк внутри Celery-задачи, создадим специализированный сервис. Он будет учитывать контекст (текст это или голос), уровень уверенности ASR и флаги мошенничества.

**Файл: `app/integrations/amocrm_formatter.py`**
```python
from app.core.models import IncomingMessage
from app.core.config import settings

def format_operator_note(msg: IncomingMessage) -> str:
    """
    Формирует структурированное, легко читаемое примечание для карточки лида в amoCRM.
    Использует Markdown-подобный синтаксис, который amoCRM корректно отображает.
    """
    channel_name = msg.channel.value.upper()
    lang_name = msg.detected_lang.upper() if msg.detected_lang else "UNKNOWN"
    
    # 1. Заголовок с мета-информацией
    note = f"🌐 **[AI Медиатор]** Канал: `{channel_name}` | Язык: `{lang_name}`\n"
    
    # 2. Предупреждения (Alerts)
    alerts = []
    
    # Проверка на мошенничество (если данные были обогащены детектором)
    fraud_score = msg.raw_payload.get("fraud_score", 0.0)
    if fraud_score >= settings.fraud_score_threshold:
        alerts.append("🚨 **ВНИМАНИЕ: Высокая вероятность мошенничества!** Проверьте запрос перед действием.")
        
    # Проверка уверенности ASR (для голосовых сообщений)
    if msg.media_type:
        asr_conf = msg.raw_payload.get("asr_confidence", 1.0)
        if asr_conf < settings.min_asr_confidence:
            alerts.append(f"⚠️ **Низкая уверенность распознавания речи ({asr_conf:.2f}).** Текст может содержать ошибки. Рекомендуется уточнить у клиента.")
            
    if alerts:
        note += "\n" + "\n".join(alerts) + "\n"
        
    note += "\n" # Отступ для читаемости

    # 3. Основной перевод (самая важная информация)
    note += f"📝 **Перевод для оператора:**\n{msg.translated_text or '[Текст отсутствует]'}\n\n"
    
    # 4. Разделитель и оригинал (для аудита и проверки)
    note += "---\n"
    note += f"🔒 **Оригинал (с маскированием PII):**\n`{msg.masked_text or msg.text or '[Пусто]'}`\n"
    
    # 5. Ссылка на полные данные (опционально, если есть S3 ключ аудио)
    if msg.audio_s3_key:
        # В реальном продакшене здесь был бы presigned URL, но для краткости оставим ключ
        note += f"\n📎 **Аудиофайл:** `{msg.audio_s3_key}`\n"

    return note.strip()

def generate_amo_tags(msg: IncomingMessage) -> list[str]:
    """
    Генерирует список тегов для автоматической категоризации лида.
    """
    tags = []
    
    # Тег канала
    tags.append(f"channel_{msg.channel.value}")
    
    # Тег языка
    if msg.detected_lang and msg.detected_lang != "unknown":
        tags.append(f"lang_{msg.detected_lang}")
        
    # Тег типа медиа
    if msg.media_type:
        tags.append(f"media_{msg.media_type.value}")
        
    # Тег мошенничества (для быстрой фильтрации в amoCRM)
    if msg.raw_payload.get("fraud_score", 0.0) >= settings.fraud_score_threshold:
        tags.append("⚠️_fraud_alert")
        
    return list(set(tags)) # Убираем дубликаты
```

---

## Шаг 3.3.2: Интеграция форматтера в Celery Pipeline

Теперь обновим наши задачи, чтобы они использовали этот форматтер при взаимодействии с amoCRM.

**Обновите файл: `app/workers/translation_tasks.py`**
```python
# ... предыдущие импорты ...
from app.integrations.amocrm_formatter import format_operator_note, generate_amo_tags

# ... внутри задачи process_incoming_message (замените блок интеграции с amoCRM на следующий) ...

            # 6. Интеграция с amoCRM с использованием форматтера
            try:
                lead_id = asyncio.run(amocrm_client.find_or_create_lead(
                    user_id=msg.user_id,
                    channel=msg.channel,
                    user_display_name=msg.user_display_name
                ))
                
                # Формируем красивое примечание
                operator_note = format_operator_note(msg)
                asyncio.run(amocrm_client.add_note(lead_id, operator_note))
                
                # Обновляем теги
                tags = generate_amo_tags(msg)
                if tags:
                    asyncio.run(amocrm_client.update_tags(lead_id, tags))
                
                logger.info("amocrm_integration_successful", lead_id=lead_id, message_id=msg.message_id, tags=tags)
                
            except pybreaker.CircuitBreakerError:
                logger.error("amocrm_circuit_open_skipping", message_id=msg.message_id)
            except Exception as e:
                logger.error("amocrm_integration_failed", message_id=msg.message_id, error=str(e), exc_info=True)

        logger.info("processing_message_completed", message_id=msg.message_id)
        return msg.model_dump()
```
*(Аналогично обновите блок интеграции с amoCRM в задаче `process_voice_message`)*

---

## Шаг 3.3.3: Исчерпывающее тестирование форматтера

Проверим, что форматтер корректно обрабатывает различные сценарии: обычный текст, голосовое сообщение с низкой уверенностью и сообщение с флагом мошенничества.

**Файл: `tests/test_amocrm_formatter.py`**
```python
import pytest
from app.core.models import IncomingMessage, Channel, MediaType
from app.integrations.amocrm_formatter import format_operator_note, generate_amo_tags
from app.core.config import settings

def test_format_operator_note_standard_text():
    msg = IncomingMessage(
        channel=Channel.MAX,
        user_id="u123",
        chat_id="c123",
        message_id="m123",
        detected_lang="uz",
        text="Salom",
        masked_text="Salom",
        translated_text="Здравствуйте"
    )
    
    note = format_operator_note(msg)
    
    assert "🌐 **[AI Медиатор]** Канал: `MAX` | Язык: `UZ`" in note
    assert "📝 **Перевод для оператора:**\nЗдравствуйте" in note
    assert "🔒 **Оригинал (с маскированием PII):**\n`Salom`" in note
    assert "⚠️" not in note # Нет предупреждений
    assert "🚨" not in note

def test_format_operator_note_voice_low_confidence():
    msg = IncomingMessage(
        channel=Channel.TELEGRAM,
        user_id="u456",
        chat_id="c456",
        message_id="m456",
        detected_lang="tg",
        media_type=MediaType.VOICE,
        audio_s3_key="telegram/voice/xyz.ogg",
        translated_text="Мне нужна помощь с документами",
        masked_text="Мне нужна помощь с документами",
        raw_payload={"asr_confidence": 0.45} # Ниже порога
    )
    
    note = format_operator_note(msg)
    
    assert "⚠️ **Низкая уверенность распознавания речи (0.45)**" in note
    assert "📎 **Аудиофайл:** `telegram/voice/xyz.ogg`" in note

def test_format_operator_note_fraud_alert():
    msg = IncomingMessage(
        channel=Channel.VK,
        user_id="u789",
        chat_id="c789",
        message_id="m789",
        detected_lang="ru",
        translated_text="Переведите деньги на безопасный счет",
        masked_text="Переведите деньги на безопасный счет",
        raw_payload={"fraud_score": 0.95} # Выше порога
    )
    
    note = format_operator_note(msg)
    assert "🚨 **ВНИМАНИЕ: Высокая вероятность мошенничества!**" in note

def test_generate_amo_tags():
    msg = IncomingMessage(
        channel=Channel.MAX,
        user_id="u123",
        chat_id="c123",
        message_id="m123",
        detected_lang="ky",
        media_type=MediaType.VOICE,
        raw_payload={"fraud_score": 0.9}
    )
    
    tags = generate_amo_tags(msg)
    
    assert "channel_max" in tags
    assert "lang_ky" in tags
    assert "media_voice" in tags
    assert "⚠️_fraud_alert" in tags
    assert len(tags) == 4
```

**Запуск тестов:**
```bash
poetry run pytest tests/test_amocrm_formatter.py -v
```

---

## Шаг 3.3.4: Production-нюансы UI/UX в amoCRM

1. **Длина примечания:** amoCRM корректно отображает длинные тексты, но если перевод превышает ~2000 символов, его лучше свернуть или обрезать в превью. В текущей реализации мы передаем полный текст, так как сообщения мигрантов обычно короткие.
2. **Теги как фильтры:** Теги вида `lang_uz` или `⚠️_fraud_alert` позволяют супервайзеру в amoCRM за 2 клика создать смарт-фильтр и выгрузить все подозрительные диалоги или диалоги на конкретном языке для контроля качества.
3. **Кастомные поля (Следующий уровень):** В будущем, вместо (или вместе с) примечаниями, можно выводить `msg.translated_text` в специальное текстовое поле карточки лида. Это позволит видеть перевод прямо в заголовке карточки, не открывая ленту событий. Для этого в `AmoCRMClient` нужно добавить метод `update_custom_field(lead_id, field_id, value)`.

---

## 🏁 ЗАВЕРШЕНИЕ ЭТАПА 3: Итоги и проверка критериев

Мы полностью завершили **Этап 3: Интеграция с amoCRM и Управление контекстом**. Давайте сверимся с исходными требованиями ТЗ:

| Критерий приемки из ТЗ | Реализация | Статус |
| :--- | :--- | :--- |
| Асинхронный клиент amoCRM с Circuit Breaker | Реализован через `httpx` + `pybreaker` с обработкой "скрытых" ошибок 200 OK. | ✅ Выполнено |
| Управление контекстом (Context Window) | Реализовано через `ContextManager` в Redis с TTL 24ч и ограничением до 10 реплик (FIFO). | ✅ Выполнено |
| Форматирование вывода для оператора | Создан `amocrm_formatter`, генерирующий структурированный Markdown с алертами (Fraud, Low ASR). | ✅ Выполнено |
| Идемпотентность и надежность | Ошибки CRM не роняют весь пайплайн перевода; Circuit Breaker защищает воркеры от зависания. | ✅ Выполнено |

**Архитектурные преимущества, закрепленные на этом этапе:**
1. **Разделение ответственности:** Логика AI (перевод) строго отделена от логики интеграции (amoCRM) через четкие интерфейсы и форматтер.
2. **Защита оператора:** Система не просто переводит, она *предупреждает* о рисках (мошенничество, плохое качество аудио), повышая безопасность и эффективность работы КЦ.
3. **Готовность к масштабированию:** Контекст в Redis и асинхронные вызовы гарантируют, что система выдержит рост нагрузки без деградации времени отклика.
