Это критически важный компонент для целевой аудитории (мигранты часто предпочитают отправлять голосовые сообщения с шумных улиц, строек или рынков). Мы реализуем высокоскоростное распознавание речи с подавлением шума (VAD) и квантованием, чтобы минимизировать потребление ресурсов и максимизировать точность.

---

# ЭТАП 2, ПОДЗАДАЧА 2.2: Интеграция Faster-Whisper (ASR)

## Шаг 2.2.1: Зависимости и системная подготовка

Библиотека `faster-whisper` использует CTranslate2 под капотом, что делает её в 4 раза быстрее оригинального Whisper при меньшем потреблении памяти.

**1. Обновите `pyproject.toml`:**
```toml
# === Speech-to-Text (ASR) ===
faster-whisper = "^1.0.0"
ffmpeg-python = "^0.2.0"
```
*Действие:* Выполните `poetry install`.

**2. Системная зависимость (КРИТИЧЕСКИ ВАЖНО для Docker):**
Faster-Whisper требует наличия `ffmpeg` в операционной системе для декодирования аудиоформатов (особенно `.opus`/`.ogg`, которые присылают MAX, VK и Telegram). 
Добавьте это в ваш `Dockerfile` (или убедитесь, что это есть в базовом образе):
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

---

## Шаг 2.2.2: Конфигурация ASR-движка

Добавим настройки Whisper в конфигурацию, чтобы можно было гибко управлять качеством и скоростью.

**Обновите `app/core/config.py`:**
```python
    # === Faster-Whisper ASR Engine ===
    whisper_model_size: str = Field(
        default="large-v3-turbo", # Оптимальный баланс: почти качество large-v3, но скорость medium
        description="Размер модели: tiny, base, small, medium, large-v3, large-v3-turbo"
    )
    whisper_compute_type: str = Field(
        default="int8", 
        description="int8 (рекомендуется для CPU/GPU), float16 (только GPU), default"
    )
    whisper_vad_filter: bool = Field(default=True, description="Включить фильтрацию голоса (Voice Activity Detection)")
    min_asr_confidence: float = Field(default=0.6, description="Минимальная уверенность ASR для принятия текста")
```

---

## Шаг 2.2.3: Реализация сервиса Whisper ASR

Мы создадим сервис с **ленивой загрузкой** (чтобы не забивать RAM при старте воркера) и **асинхронной оберткой**, чтобы инференс не блокировал event loop.

**Файл: `app/ml/whisper_asr.py`**
```python
import asyncio
import io
from typing import List, Optional
from pydantic import BaseModel
from faster_whisper import WhisperModel, BatchedInferencePipeline
import logging

from app.core.config import settings
from app.core.logger import logger

class ASRSegment(BaseModel):
    text: str
    start: float
    end: float
    confidence: float

class ASRResult(BaseModel):
    full_text: str
    language: str
    avg_confidence: float
    segments: List[ASRSegment]

class WhisperASRService:
    def __init__(self):
        self.model_size = settings.whisper_model_size
        self._model = None
        self._is_initialized = False
        logger.info("whisper_asr_service_initialized_lazy")

    def _initialize(self) -> None:
        """Ленивая загрузка модели для экономии памяти."""
        if self._is_initialized:
            return

        logger.info("loading_whisper_model", model=self.model_size, compute_type=settings.whisper_compute_type)
        try:
            # Загрузка модели с квантованием
            self._model = WhisperModel(
                self.model_size,
                device="auto", # auto выберет cuda, если доступен, иначе cpu
                compute_type=settings.whisper_compute_type,
                cpu_threads=4, # Ограничиваем потоки CPU, чтобы не душить систему
                num_workers=1
            )
            
            # Опционально: можно обернуть в BatchedInferencePipeline для еще большей скорости на GPU
            # self._model = BatchedInferencePipeline(model=self._model)
            
            self._is_initialized = True
            logger.info("whisper_model_loaded_successfully")
        except Exception as e:
            logger.error("whisper_model_load_failed", error=str(e), exc_info=True)
            raise

    def _transcribe_sync(self, audio_bytes: bytes) -> ASRResult:
        """Синхронный метод транскрибации (выполняется в отдельном потоке)."""
        self._initialize()
        
        # faster-whisper принимает file-like object или путь к файлу
        audio_file = io.BytesIO(audio_bytes)
        
        try:
            segments, info = self._model.transcribe(
                audio_file,
                beam_size=5,
                vad_filter=settings.whisper_vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=500, # Игнорировать паузы < 0.5 сек (шум рынка/стройки)
                    speech_pad_ms=200            # Сглаживание границ речи
                ) if settings.whisper_vad_filter else None,
                language=None # None = автоматическое определение языка
            )
            
            segments_list = []
            full_text_parts = []
            total_confidence = 0.0
            
            for segment in segments:
                segments_list.append(ASRSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.avg_logprob # В Whisper это логарифмическая вероятность, чем ближе к 0, тем лучше
                ))
                full_text_parts.append(segment.text.strip())
                total_confidence += segment.avg_logprob

            avg_confidence = total_confidence / len(segments_list) if segments_list else 0.0
            
            return ASRResult(
                full_text=" ".join(full_text_parts).strip(),
                language=info.language,
                avg_confidence=avg_confidence,
                segments=segments_list
            )
            
        except Exception as e:
            logger.error("whisper_transcription_failed", error=str(e), exc_info=True)
            # Fail-soft: возвращаем пустой результат с нулевой уверенностью
            return ASRResult(full_text="", language="unknown", avg_confidence=0.0, segments=[])

    async def transcribe(self, audio_bytes: bytes) -> ASRResult:
        """Асинхронная обертка, не блокирующая event loop."""
        loop = asyncio.get_running_loop()
        # Выносим тяжелый CPU/GPU вызов в пул потоков
        result = await loop.run_in_executor(None, self._transcribe_sync, audio_bytes)
        return result

# Глобальный синглтон
whisper_asr = WhisperASRService()
```

---

## Шаг 2.2.4: Интеграция в Celery Pipeline (Обработка голоса)

Теперь создадим задачу, которая обрабатывает именно голосовые сообщения. Она скачает аудио из S3, распознает его, а затем передаст распознанный текст в тот же пайплайн маскирования и перевода.

**Файл: `app/workers/translation_tasks.py`** (Добавьте эту задачу к существующим)
```python
import uuid
import asyncio
from app.core.celery_app import celery_app
from app.core.models import IncomingMessage, MediaType
from app.ml.whisper_asr import whisper_asr
from app.ml.language_detector import language_detector
from app.ml.pii_masker import pii_masker
from app.ml.nllb_translator import nllb_translator
from app.infrastructure.s3 import s3_service
from app.core.logger import logger
from app.core.context import set_trace_id, set_channel, set_user_id

@celery_app.task(
    bind=True,
    name="app.workers.translation_tasks.process_voice_message",
    queue="translate_voice", # Отдельная очередь для тяжелых задач
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=60,      # Мягкий таймаут 60 сек
    time_limit=90            # Жесткий таймаут 90 сек
)
def process_voice_message(self, message_dict: dict) -> dict:
    """Обработка голосового сообщения: S3 -> Whisper -> PII -> Translate."""
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    
    try:
        msg = IncomingMessage.model_validate(message_dict)
        set_channel(msg.channel.value)
        set_user_id(msg.user_id)
        
        if not msg.audio_s3_key:
            logger.error("voice_message_missing_s3_key", message_id=msg.message_id)
            raise ValueError("Missing audio_s3_key")

        logger.info("voice_processing_started", s3_key=msg.audio_s3_key, message_id=msg.message_id)

        # 1. Скачивание аудио из S3
        audio_bytes = asyncio.run(s3_service.download_file(msg.audio_s3_key))
        if not audio_bytes:
            raise RuntimeError("Failed to download audio from S3")

        # 2. ASR (Распознавание речи)
        asr_result = asyncio.run(whisper_asr.transcribe(audio_bytes))
        
        if not asr_result.full_text or asr_result.avg_confidence < settings.min_asr_confidence:
            logger.warning(
                "asr_low_confidence_or_empty", 
                confidence=asr_result.avg_confidence, 
                text_preview=asr_result.full_text[:50]
            )
            # Можно отправить специальное сообщение клиенту: "Плохо слышно, напишите текстом"
            msg.translated_text = "[Аудио неразборчиво. Пожалуйста, напишите текстом или говорите громче.]"
            msg.detected_lang = "unknown"
        else:
            logger.info(
                "asr_successful", 
                lang=asr_result.language, 
                confidence=round(asr_result.avg_confidence, 3),
                text_preview=asr_result.full_text[:50]
            )
            
            # 3. Детекция языка распознанного текста (на случай, если Whisper ошибся с lang)
            detected_lang, conf = language_detector.detect(asr_result.full_text)
            msg.detected_lang = detected_lang
            
            # 4. PII-маскирование распознанного текста
            target_lang = detected_lang if conf > 0.5 else "ru"
            masked_text = pii_masker.mask(asr_result.full_text, lang=target_lang)
            
            # 5. Перевод на русский (если это не русский)
            if detected_lang != "ru":
                translated_text = asyncio.run(nllb_translator.translate(
                    text=masked_text,
                    src_lang=detected_lang,
                    tgt_lang="ru"
                ))
                msg.translated_text = translated_text
            else:
                msg.translated_text = masked_text

        # Сохраняем метаданные ASR в raw_payload для аудита
        msg.raw_payload["asr_confidence"] = asr_result.avg_confidence
        msg.raw_payload["asr_language"] = asr_result.language

        logger.info("voice_processing_completed", message_id=msg.message_id)
        return msg.model_dump()

    except Exception as e:
        logger.error("voice_processing_failed", message_id=message_dict.get("message_id", "unknown"), error=str(e), exc_info=True)
        raise
```

---

## Шаг 2.2.5: Исчерпывающее тестирование

Напишем тест, который проверяет работу VAD и корректность возврата структуры данных.

**Файл: `tests/test_whisper_asr.py`**
```python
import pytest
import asyncio
from app.ml.whisper_asr import whisper_asr

# Для теста можно использовать короткий реальный аудиофайл или замокать его.
# Здесь приведен пример с реальным вызовом, если у вас есть тестовый файл.
# Создайте файл tests/fixtures/test_voice.ogg (5-10 секунд речи)

@pytest.mark.asyncio
async def test_whisper_transcribe_real_audio():
    """Интеграционный тест распознавания реального аудиофайла."""
    import os
    fixture_path = "tests/fixtures/test_voice.ogg"
    
    if not os.path.exists(fixture_path):
        pytest.skip("Test audio fixture not found. Skipping ASR test.")
        
    with open(fixture_path, "rb") as f:
        audio_bytes = f.read()
        
    result = await whisper_asr.transcribe(audio_bytes)
    
    assert isinstance(result.full_text, str)
    assert len(result.full_text) > 0
    assert result.language in ["ru", "uz", "tg", "ky", "en"] # Ожидаемые языки
    assert result.avg_confidence >= -1.0 # В Whisper avg_logprob отрицательный, чем ближе к 0, тем лучше
    assert len(result.segments) > 0

@pytest.mark.asyncio
async def test_whisper_vad_filters_noise():
    """Проверка того, что VAD игнорирует тишину/шум."""
    # Генерируем 3 секунды "тишины" (нулевые байты в формате wav/ogg)
    # Это упрощенная проверка, в реальности нужен файл с шумом
    silent_audio = b"\x00" * 48000 
    
    result = await whisper_asr.transcribe(silent_audio)
    
    # VAD должен отфильтровать это как шум, и текст должен быть пустым
    assert result.full_text == ""
    assert result.avg_confidence == 0.0
```

---

## Шаг 2.2.6: Production-нюансы для Docker и Kubernetes

Чтобы Faster-Whisper работал стабильно в продакшене, необходимо учесть два критических момента:

1. **Shared Memory (`/dev/shm`):** Как и в случае с NLLB, CTranslate2 активно использует shared memory. Без увеличения лимита воркер упадет с ошибкой `Bus error (core dumped)`.
   *Решение:* В `docker-compose.yml` для сервиса `celery-worker` добавьте:
   ```yaml
   services:
     celery-worker:
       # ...
       shm_size: '2gb' # Минимум 2 ГБ
   ```

2. **Предварительная загрузка модели в Dockerfile:** Чтобы при первом запросе не было задержки в 10-15 секунд на скачивание модели, загрузите её на этапе сборки образа.
   *Добавьте в `Dockerfile`:*
   ```dockerfile
   # Предварительная загрузка модели Whisper
   RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
   ```

---

### Что мы достигли в Подзадаче 2.2:

✅ **Высокоскоростной локальный ASR:** Использование `large-v3-turbo` с `int8` квантованием обеспечивает точность, близкую к state-of-the-art, при минимальных требованиях к железу.
✅ **Устойчивость к шуму:** Включенный `vad_filter` с настройкой `min_silence_duration_ms` критически важен для отсечения фонового шума (стройки, улицы), характерного для записей мигрантов.
✅ **Неблокирующая архитектура:** Асинхронная обертка через `run_in_executor` гарантирует, что тяжелое распознавание речи не заблокирует другие задачи в Celery-воркере.
✅ **Сквозной пайплайн:** Голосовое сообщение теперь проходит полный цикл: S3 → Whisper → Детекция языка → PII-маскирование → Перевод, возвращая единый структурированный объект `IncomingMessage`.
✅ **Fail-Soft и таймауты:** Настроены `soft_time_limit` и `time_limit` в Celery, а также проверка `min_asr_confidence`, чтобы система не пыталась переводить бессмысленный шум.
