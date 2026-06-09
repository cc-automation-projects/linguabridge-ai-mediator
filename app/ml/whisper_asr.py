import asyncio
import io

from faster_whisper import WhisperModel
from pydantic import BaseModel

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
    segments: list[ASRSegment]


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
