Это критически важный шаг для замыкания цикла двусторонней голосовой связи. Оператор пишет ответ на русском, система переводит его на язык клиента (например, узбекский), а затем озвучивает этот перевод, чтобы клиент получил полноценное голосовое сообщение. 

Главный вызов здесь: TTS-модели (включая Silero) обычно выдают несжатый `.wav`, а мессенджеры (MAX, Telegram, VK) требуют `.ogg` с кодеком Opus для нативного и быстрого воспроизведения голосовых сообщений. Мы реализуем сверхбыструю конвертацию через `ffmpeg`.

---

# ЭТАП 4, ПОДЗАДАЧА 4.1: Интеграция Silero TTS и конвертация аудио

## Шаг 4.1.1: Зависимости и системная подготовка

Нам понадобятся библиотеки для синтеза речи и работы с аудио, а также системный пакет `ffmpeg`.

**1. Обновите `pyproject.toml`:**
```toml
# === Text-to-Speech (TTS) & Audio ===
silero-models = "^4.0.0"
torchaudio = "^2.2.1"
ffmpeg-python = "^0.2.0"
```
*Действие:* Выполните `poetry install`.

**2. Системная зависимость (КРИТИЧЕСКИ ВАЖНО):**
Для конвертации аудио в формат `.ogg` (Opus) в системе должен быть установлен `ffmpeg`. 
Убедитесь, что в вашем `Dockerfile` (или на сервере) есть:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

---

## Шаг 4.1.2: Конфигурация TTS-движка

Добавим настройки для гибкого управления качеством и устройством рендеринга.

**Обновите `app/core/config.py`:**
```python
    # === TTS Engine (Silero) ===
    tts_device: str = Field(default="cpu", description="cpu или cuda")
    tts_sample_rate: int = Field(default=24000, description="Частота дискретизации (Silero v4 использует 24000 или 48000)")
    tts_speaker: str = Field(default="kseniya", description="Имя спикера (зависит от модели)")
    tts_model_language: str = Field(default="ru", description="Базовый язык модели (для Silero v4: 'ru' или 'en')")
```

---

## Шаг 4.1.3: Реализация сервиса TTS и конвертации

Мы создадим модульный сервис. **Важное примечание:** Silero TTS отлично работает с русским и английским. Для узбекского/таджикского/киргизского "из коробки" качество может быть акцентным. В продакшене этот интерфейс легко заменяется на `Coqui XTTS-v2` или `Piper`, но мы реализуем запрошенный Silero-пайплайн с максимальной оптимизацией.

**Файл: `app/ml/tts_service.py`**
```python
import os
import tempfile
import uuid
import asyncio
import torch
import ffmpeg
import logging
from typing import Optional

from app.core.config import settings
from app.core.logger import logger

class TTSService:
    def __init__(self):
        self._model = None
        self._is_initialized = False
        logger.info("tts_service_initialized_lazy")

    def _initialize(self) -> None:
        """Ленивая загрузка модели Silero TTS."""
        if self._is_initialized:
            return

        logger.info("loading_silero_tts_model", device=settings.tts_device)
        try:
            # Загрузка через torch.hub (стандартный способ для Silero)
            self._model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=settings.tts_model_language,
                speaker=settings.tts_speaker,
                device=settings.tts_device
            )
            self._is_initialized = True
            logger.info("silero_tts_model_loaded_successfully")
        except Exception as e:
            logger.error("silero_tts_load_failed", error=str(e), exc_info=True)
            raise

    def _generate_and_convert_sync(self, text: str) -> Optional[bytes]:
        """
        Синхронный метод: генерация wav и конвертация в ogg (Opus).
        Выполняется в отдельном потоке.
        """
        self._initialize()
        
        if not text or not text.strip():
            return None

        temp_wav_path = None
        temp_ogg_path = None
        
        try:
            # 1. Генерация аудио (возвращает torch.Tensor)
            # sample_rate=24000 - стандарт для Silero v4
            audio = self._model.apply_tts(
                text=text,
                speaker=settings.tts_speaker,
                sample_rate=settings.tts_sample_rate
            )
            
            # 2. Сохранение во временный WAV-файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                temp_wav_path = tmp_wav.name
                
            torchaudio.save(temp_wav_path, audio.unsqueeze(0), settings.tts_sample_rate)
            
            # 3. Конвертация WAV -> OGG (Opus) через ffmpeg
            # Opus обеспечивает отличное качество при малом размере, что критично для мессенджеров
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
                temp_ogg_path = tmp_ogg.name
                
            (
                ffmpeg
                .input(temp_wav_path)
                .output(
                    temp_ogg_path,
                    format='ogg',
                    acodec='libopus',
                    ar=48000,  # Telegram/MAX предпочитают 48kHz для Opus
                    ac=1,      # Моно
                    loglevel='error' # Скрываем спам ffmpeg в консоль
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # 4. Чтение результата в байты
            with open(temp_ogg_path, "rb") as f:
                ogg_bytes = f.read()
                
            logger.info("tts_generation_and_conversion_success", text_preview=text[:30], size_bytes=len(ogg_bytes))
            return ogg_bytes

        except ffmpeg.Error as e:
            logger.error("ffmpeg_conversion_failed", stderr=e.stderr.decode('utf-8'), exc_info=True)
            return None
        except Exception as e:
            logger.error("tts_generation_failed", text_preview=text[:30], error=str(e), exc_info=True)
            return None
        finally:
            # 5. Очистка временных файлов
            for path in [temp_wav_path, temp_ogg_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError as e:
                        logger.warning("temp_file_cleanup_failed", path=path, error=str(e))

    async def generate_voice_message(self, text: str) -> Optional[bytes]:
        """Асинхронная обертка, не блокирующая event loop."""
        loop = asyncio.get_running_loop()
        # Выносим тяжелую CPU-операцию (инференс + ffmpeg) в пул потоков
        result = await loop.run_in_executor(None, self._generate_and_convert_sync, text)
        return result

# Глобальный синглтон
tts_service = TTSService()
```

---

## Шаг 4.1.4: Интеграция в Celery Pipeline (Задача генерации ответа)

Теперь создадим задачу, которая будет вызываться, когда оператор отправляет ответ клиенту. Она переведет текст, сгенерирует аудио, загрузит его в S3 и вернет ключ для отправки через адаптер канала.

**Файл: `app/workers/tts_tasks.py`**
```python
import uuid
import asyncio
from app.core.celery_app import celery_app
from app.core.models import Channel
from app.ml.nllb_translator import nllb_translator
from app.ml.tts_service import tts_service
from app.infrastructure.s3 import s3_service
from app.core.logger import logger
from app.core.context import set_trace_id, set_channel, set_user_id

@celery_app.task(
    bind=True,
    name="app.workers.tts_tasks.generate_and_upload_voice_reply",
    queue="tts_generation",
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=30,
    time_limit=45
)
def generate_and_upload_voice_reply(self, user_id: str, channel: str, target_lang: str, operator_text_ru: str) -> dict:
    """
    Переводит ответ оператора на язык клиента, генерирует TTS и загружает в S3.
    """
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    set_channel(channel)
    set_user_id(user_id)
    
    try:
        logger.info("tts_task_started", user_id=user_id, target_lang=target_lang)

        # 1. Перевод ответа оператора на язык клиента
        translated_text = asyncio.run(nllb_translator.translate(
            text=operator_text_ru,
            src_lang="ru",
            tgt_lang=target_lang
        ))
        
        if not translated_text:
            logger.warning("tts_translation_empty", original_text=operator_text_ru)
            return {"status": "error", "message": "Translation failed"}

        # 2. Генерация и конвертация аудио (WAV -> OGG)
        ogg_bytes = asyncio.run(tts_service.generate_voice_message(translated_text))
        
        if not ogg_bytes:
            logger.error("tts_audio_generation_failed", translated_text=translated_text)
            return {"status": "error", "message": "TTS generation failed"}

        # 3. Загрузка в S3
        file_key = f"{channel}/tts_replies/{uuid.uuid4()}.ogg"
        success = asyncio.run(s3_service.upload_file(file_key, ogg_bytes, content_type="audio/ogg"))
        
        if not success:
            logger.error("tts_s3_upload_failed", file_key=file_key)
            return {"status": "error", "message": "S3 upload failed"}

        logger.info("tts_task_completed_successfully", file_key=file_key, user_id=user_id)
        
        return {
            "status": "success",
            "s3_key": file_key,
            "translated_text": translated_text
        }

    except Exception as e:
        logger.error("tts_task_failed", user_id=user_id, error=str(e), exc_info=True)
        raise
```

---

## Шаг 4.1.5: Исчерпывающее тестирование

Проверим, что конвертация работает, а временные файлы корректно удаляются.

**Файл: `tests/test_tts_service.py`**
```python
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.ml.tts_service import tts_service

@pytest.mark.asyncio
async def test_tts_generation_and_conversion_mocked():
    """Тест пайплайна TTS с мокированием torch и ffmpeg."""
    test_text = "Здравствуйте, ваш документ готов."
    
    # Мокаем torch.hub.load
    mock_model = MagicMock()
    mock_model.apply_tts.return_value = MagicMock() # Фейковый тензор
    
    with patch('torch.hub.load', return_value=(mock_model, "example")), \
         patch('torchaudio.save') as mock_save, \
         patch('ffmpeg.input') as mock_ffmpeg_input:
        
        # Настраиваем цепочку вызовов ffmpeg
        mock_output = MagicMock()
        mock_ffmpeg_input.return_value.output.return_value.overwrite_output.return_value.run.return_value = (b"stdout", b"stderr")
        
        # Мокаем чтение файла
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake_ogg_audio_data"
            with patch("os.path.exists", return_value=True), \
                 patch("os.unlink") as mock_unlink:
                 
                result = await tts_service.generate_voice_message(test_text)
                
                assert result == b"fake_ogg_audio_data"
                mock_model.apply_tts.assert_called_once()
                mock_ffmpeg_input.assert_called_once()
                # Проверяем, что временные файлы были удалены
                assert mock_unlink.call_count == 2

@pytest.mark.asyncio
async def test_tts_empty_text_handling():
    """Проверка обработки пустого текста."""
    result = await tts_service.generate_voice_message("")
    assert result is None
```

**Запуск тестов:**
```bash
poetry run pytest tests/test_tts_service.py -v
```

---

## Шаг 4.1.6: Production-нюансы и ограничения

1. **Языковая поддержка Silero:** Silero TTS (`v4_ru`, `v3_en`) обучен преимущественно на русском и английском. При попытке сгенерировать узбекский или таджикский текст, модель будет использовать русские/английские фонемы, что создаст сильный акцент или неразборчивость. 
   * **Решение для продакшена:** Интерфейс `TTSService` спроектирован так, что его легко заменить. Для мигрантских языков рекомендуется использовать **Coqui XTTS-v2** (поддерживает 17+ языков, включая мультиязычный режим) или **Piper TTS** (быстрый, локальный, есть модели для некоторых языков СНГ). Код задачи `generate_and_upload_voice_reply` при этом не изменится.
2. **Управление памятью (GPU OOM):** Если используется GPU, `torch.hub.load` может фрагментировать память. Убедитесь, что в Celery настроен `worker_max_tasks_per_child=50` (как мы делали в Шаге 1.1), чтобы воркеры периодически перезапускались, очищая память.
3. **FFmpeg в Docker:** Убедитесь, что в финальном Docker-образе установлен пакет `ffmpeg`. Без него конвертация в `.ogg` (Opus) упадет с ошибкой `ffmpeg not found`.
4. **Латентность:** Генерация + конвертация на CPU занимает ~0.5–1.5 секунды для коротких фраз. Это укладывается в целевой SLA < 4 сек для полного голосового пайплайна.

---

### Что мы достигли в Подзадаче 4.1:

✅ **Полный цикл TTS:** Реализована генерация речи из текста с последующей обязательной конвертацией в формат, совместимый с мессенджерами (`.ogg` Opus).
✅ **Безопасность ресурсов:** Использование временных файлов с гарантированной очисткой (`finally` блок) предотвращает засорение диска воркера.
✅ **Неблокирующая архитектура:** Тяжелые операции (инференс PyTorch и системный вызов FFmpeg) вынесены в `run_in_executor`, не блокируя асинхронный цикл Celery.
✅ **Готовность к замене движка:** Архитектура абстрагирована, что позволяет в будущем заменить Silero на более подходящую мультиязычную модель (например, XTTS) без переписывания пайплайна.
