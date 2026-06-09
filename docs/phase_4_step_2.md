Это критически важный шаг для защиты уязвимой категории клиентов (мигрантов) и повышения точности распознавания речи в реальных, зашумленных условиях (стройка, улица, рынок). Мы реализуем сверхбыстрый детектор мошеннических паттернов и программную очистку аудио перед подачей в Whisper.

---

# ЭТАП 4, ПОДЗАДАЧА 4.2: Fraud Detector и шумоподавление аудио

## Шаг 4.2.1: Зависимости и конфигурация

Мы повторно используем уже установленную библиотеку `pyahocorasick` для молниеносного поиска множества паттернов мошенничества за один проход, и `ffmpeg-python` для фильтрации аудио. Новых зависимостей не требуется.

**Обновите `app/core/config.py`:**
```python
    # === Fraud Detection ===
    fraud_score_threshold: float = Field(default=0.7, description="Порог срабатывания алерта о мошенничестве (0.0 - 1.0)")
    
    # === Audio Pre-processing ===
    audio_enable_noise_reduction: bool = Field(default=True, description="Включить фильтрацию шума перед ASR")
    audio_highpass_freq: int = Field(default=200, description="Частота среза низких частот (удаление гула)")
    audio_lowpass_freq: int = Field(default=8000, description="Частота среза высоких частот (удаление шипения)")
```

---

## Шаг 4.2.2: Реализация эвристического Fraud Detector

Мы используем алгоритм Aho-Corasick (как и в Terminology Override), но с весами для каждого паттерна. Это позволяет рассчитать итоговый "скор риска" и вернуть список сработавших триггеров.

**Файл: `app/ml/fraud_detector.py`**
```python
import ahocorasick
import logging
from typing import List, Tuple
from app.core.config import settings
from app.core.logger import logger

class FraudDetectorService:
    def __init__(self):
        self._automaton = None
        self._patterns_weights = {}
        self._is_initialized = False
        logger.info("fraud_detector_service_initialized_lazy")

    def _initialize(self) -> None:
        if self._is_initialized:
            return

        # Список паттернов и их "вес" опасности (от 0.1 до 1.0)
        patterns = [
            ("безопасный счет", 0.9),
            ("сотрудник полиции", 0.8),
            ("служба безопасности банка", 0.8),
            ("переведите деньги", 0.7),
            ("код из смс", 0.9),
            ("никому не говорите", 0.8),
            ("решить вопрос с документами", 0.6),
            ("взяли кредит на ваше имя", 0.8),
            ("демонстрация экрана", 0.7),
            ("удалите приложение", 0.9),
        ]

        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY)
        for pattern, weight in patterns:
            # Добавляем в нижнем регистре для нечувствительного к регистру поиска
            self._automaton.add_word(pattern.lower(), (pattern, weight))
            
        self._automaton.make_automaton()
        self._is_initialized = True
        logger.info("fraud_detector_automaton_built", patterns_count=len(patterns))

    def check_text(self, text: str) -> Tuple[float, List[str]]:
        """
        Анализирует текст на наличие мошеннических паттернов.
        :return: Кортеж (общий скор риска от 0.0 до 1.0, список сработавших паттернов)
        """
        if not text or not text.strip():
            return 0.0, []

        self._initialize()
        
        text_lower = text.lower()
        matches = list(self._automaton.iter(text_lower))
        
        if not matches:
            return 0.0, []

        triggered_patterns = set()
        max_weight = 0.0
        
        # Агрегируем сработавшие паттерны. 
        # Используем макс. вес как базовый скор, так как один серьезный триггер важнее нескольких мелких.
        for end_index, (pattern, weight) in matches:
            triggered_patterns.add(pattern)
            if weight > max_weight:
                max_weight = weight

        # Нормализуем скор: если сработало несколько паттернов, увеличиваем скор, но не выше 1.0
        final_score = min(1.0, max_weight + (len(triggered_patterns) - 1) * 0.1)
        
        return round(final_score, 2), list(triggered_patterns)

# Глобальный синглтон
fraud_detector = FraudDetectorService()
```

---

## Шаг 4.2.3: Предобработка аудио (Шумоподавление)

Вместо тяжелых ML-моделей для шумоподавления (которые требуют много CPU/GPU), мы используем встроенные, высокооптимизированные фильтры FFmpeg. Полосовой фильтр (Bandpass) отсекает низкочастотный гул (транспорт, ветер) и высокочастотное шипение, оставляя только диапазон человеческой речи (200 Гц – 8000 Гц).

**Файл: `app/ml/audio_utils.py`**
```python
import io
import ffmpeg
import logging
from app.core.config import settings
from app.core.logger import logger

def reduce_audio_noise(audio_bytes: bytes) -> bytes:
    """
    Применяет полосовую фильтрацию (Bandpass) для удаления фонового шума 
    и выделения диапазона человеческой речи перед подачей в ASR.
    """
    if not settings.audio_enable_noise_reduction:
        return audio_bytes

    try:
        # Читаем исходные байты как поток
        input_stream = ffmpeg.input('pipe:0', format='ogg') # Предполагаем, что на входе уже ogg/opus
        
        # Применяем фильтры:
        # highpass: удаляет низкочастотный гул (ветер, моторы)
        # lowpass: удаляет высокочастотное шипение
        # afftdn: легкое адаптивное подавление шума (nf=-15 - мягкий режим, чтобы не исказить голос)
        filtered_stream = (
            input_stream
            .filter('highpass', f=settings.audio_highpass_freq)
            .filter('lowpass', f=settings.audio_lowpass_freq)
            .filter('afftdn', nf=-15)
            .output('pipe:1', format='ogg', acodec='libopus')
        )
        
        # Выполняем конвертацию, передавая исходные байты через stdin
        out_bytes, _ = filtered_stream.run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
        
        logger.debug("audio_noise_reduction_applied", original_size=len(audio_bytes), filtered_size=len(out_bytes))
        return out_bytes
        
    except ffmpeg.Error as e:
        # FAIL-SOFT: Если фильтрация падает, возвращаем исходное аудио, чтобы не ломать пайплайн
        logger.warning("audio_filtering_failed_fallback_to_original", stderr=e.stderr.decode('utf-8', errors='ignore'))
        return audio_bytes
    except Exception as e:
        logger.error("audio_processing_unexpected_error", error=str(e))
        return audio_bytes
```

---

## Шаг 4.2.4: Интеграция в Celery Pipeline

Теперь мы встраиваем эти два новых шага в задачу обработки голосовых сообщений. Порядок критически важен: **Скачивание → Шумоподавление → ASR → Перевод → Fraud Check**.

**Обновите файл: `app/workers/translation_tasks.py`**
```python
# ... предыдущие импорты ...
from app.ml.audio_utils import reduce_audio_noise
from app.ml.fraud_detector import fraud_detector

# ... внутри задачи process_voice_message ...

        # 1. Скачивание аудио из S3
        audio_bytes = asyncio.run(s3_service.download_file(msg.audio_s3_key))
        if not audio_bytes:
            raise RuntimeError("Failed to download audio from S3")

        # 2. Предобработка аудио (Шумоподавление)
        cleaned_audio_bytes = reduce_audio_noise(audio_bytes)

        # 3. ASR (Распознавание речи) на очищенном аудио
        asr_result = asyncio.run(whisper_asr.transcribe(cleaned_audio_bytes))
        
        if not asr_result.full_text or asr_result.avg_confidence < settings.min_asr_confidence:
            logger.warning("asr_low_confidence_or_empty", confidence=asr_result.avg_confidence)
            msg.translated_text = "[Аудио неразборчиво. Пожалуйста, напишите текстом или говорите громче.]"
            msg.detected_lang = "unknown"
        else:
            # 4. Детекция языка и PII-маскирование
            detected_lang, conf = language_detector.detect(asr_result.full_text)
            msg.detected_lang = detected_lang
            target_lang = detected_lang if conf > 0.5 else "ru"
            masked_text = pii_masker.mask(asr_result.full_text, lang=target_lang)
            
            # 5. Перевод на русский
            if detected_lang != "ru":
                translated_text = asyncio.run(nllb_translator.translate(
                    text=masked_text, src_lang=detected_lang, tgt_lang="ru"
                ))
                msg.translated_text = translated_text
            else:
                msg.translated_text = masked_text

            # 6. FRAUD DETECTION (Проверяем переведенный текст, так как его читает оператор)
            fraud_score, fraud_flags = fraud_detector.check_text(msg.translated_text)
            
            # Сохраняем результаты в raw_payload для передачи в amoCRM и аудит
            msg.raw_payload["asr_confidence"] = asr_result.avg_confidence
            msg.raw_payload["asr_language"] = asr_result.language
            msg.raw_payload["fraud_score"] = fraud_score
            msg.raw_payload["fraud_flags"] = fraud_flags

            if fraud_score >= settings.fraud_score_threshold:
                logger.warning(
                    "fraud_alert_triggered", 
                    score=fraud_score, 
                    flags=fraud_flags, 
                    message_id=msg.message_id
                )

        # 7. Сохранение в контекст и интеграция с amoCRM (используя format_operator_note, который теперь учтет fraud_score)
        # ... (код из предыдущих шагов остается без изменений) ...
```

---

## Шаг 4.2.5: Исчерпывающее тестирование

Проверим, что детектор корректно считает скоры, а аудио-фильтр не ломает пайплайн при ошибках.

**Файл: `tests/test_fraud_and_audio.py`**
```python
import pytest
from app.ml.fraud_detector import fraud_detector
from app.ml.audio_utils import reduce_audio_noise
from app.core.config import settings

class TestFraudDetector:
    def test_no_fraud_detected(self):
        text = "Здравствуйте, хочу узнать график работы офиса."
        score, flags = fraud_detector.check_text(text)
        assert score == 0.0
        assert len(flags) == 0

    def test_single_fraud_pattern(self):
        text = "Мне нужно перевести деньги на безопасный счет."
        score, flags = fraud_detector.check_text(text)
        assert score >= 0.8
        assert "безопасный счет" in flags
        assert "переведите деньги" in flags

    def test_multiple_fraud_patterns_aggregation(self):
        text = "Это сотрудник полиции. Назовите код из смс и никому не говорите."
        score, flags = fraud_detector.check_text(text)
        assert score == 1.0 # Максимальный скор из-за множественных тяжелых триггеров
        assert len(flags) >= 2

    def test_case_insensitive(self):
        text = "Сотрудник Полиции просит удалить приложение."
        score, flags = fraud_detector.check_text(text)
        assert score > 0.7
        assert "сотрудник полиции" in flags

class TestAudioUtils:
    def test_noise_reduction_passthrough_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "audio_enable_noise_reduction", False)
        dummy_audio = b"fake_audio_data"
        result = reduce_audio_noise(dummy_audio)
        assert result == dummy_audio

    def test_noise_reduction_fail_soft_on_invalid_audio(self):
        # Передаем не аудио, а текст. FFmpeg должен упасть, но функция должна вернуть исходные данные
        invalid_audio = b"this is not an ogg file"
        result = reduce_audio_noise(invalid_audio)
        # Ожидаем, что функция отловит ffmpeg.Error и вернет исходные байты (fail-soft)
        assert result == invalid_audio
```

**Запуск тестов:**
```bash
poetry run pytest tests/test_fraud_and_audio.py -v
```

---

## Шаг 4.2.6: Production-нюансы

1. **Ложные срабатывания (False Positives):** Паттерн "сотрудник полиции" может сработать, если клиент *действительно* звонит, чтобы сообщить о краже документов. Поэтому Fraud Detector **никогда не блокирует** диалог автоматически. Он только выставляет `fraud_score` и добавляет флаг `⚠️_fraud_alert` в теги amoCRM (как мы настроили в `amocrm_formatter.py`), чтобы оператор был начеку.
2. **Производительность FFmpeg:** Фильтры `highpass`/`lowpass` и мягкий `afftdn` работают очень быстро на CPU. Однако, если воркер обрабатывает десятки длинных голосовых сообщений одновременно, это может создать нагрузку. Очередь `translate_voice` и `worker_prefetch_multiplier=1` (из Этапа 1) защищают от перегрузки.
3. **Язык проверки:** Мы проверяем на мошенничество *переведенный* русский текст. Это осознанное решение: оператору проще понять контекст на русском, а NLLB обычно корректно переводит ключевые термины ("safe account" -> "безопасный счет"). Для максимальной надежности в будущем можно добавить проверку и оригинального текста через мультиязычный классификатор (например, `ruRoberta-small`).

---

### Что мы достигли в Подзадаче 4.2:

✅ **Защита клиентов:** Внедрен молниеносный (O(N)) детектор социальных инженерий, который предупреждает оператора о рисках, не блокируя легитимные диалоги.  
✅ **Повышение точности ASR:** Программная очистка аудио от низкочастотного гула и шипения через FFmpeg значительно снижает WER (Word Error Rate) для записей, сделанных на улице или производстве.  
✅ **Архитектурная устойчивость (Fail-Soft):** Если фильтрация аудио по какой-то причине падает, система элегантно откатывается к исходному файлу, не прерывая весь пайплайн.  
✅ **Сквозная видимость:** Скор мошенничества и сработавшие флаги сохраняются в `raw_payload`, передаются в форматтер amoCRM и в таблицу аудита PostgreSQL для последующего анализа службой безопасности.  
