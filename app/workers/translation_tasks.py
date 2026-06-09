import asyncio
import time
import uuid

import pybreaker

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.context import set_channel, set_trace_id, set_user_id
from app.core.logger import logger
from app.core.models import ConversationTurn, IncomingMessage
from app.core.observability import fraud_alerts_counter, translation_latency_histogram
from app.infrastructure.context_manager import context_manager
from app.infrastructure.s3 import s3_service
from app.integrations.amocrm_client import amocrm_client
from app.integrations.amocrm_formatter import format_operator_note, generate_amo_tags
from app.ml.audio_preprocessor import audio_preprocessor
from app.ml.fraud_detector import fraud_detector
from app.ml.language_detector import language_detector
from app.ml.nllb_translator import nllb_translator
from app.ml.pii_masker import pii_masker
from app.ml.terminology_override import terminology_override
from app.ml.whisper_asr import whisper_asr


@celery_app.task(
    bind=True,
    name="app.workers.translation_tasks.process_incoming_message",
    queue="translate_text", # Будет переопределено динамически при вызове
    acks_late=True,
    reject_on_worker_lost=True
)
def process_incoming_message(self, message_dict: dict) -> dict:
    """
    Основная задача обработки входящего сообщения: детекция языка и PII-маскирование.
    """
    # 1. Восстановление контекста для логирования
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    start_time = time.time()

    try:
        msg = IncomingMessage.model_validate(message_dict)
        set_channel(msg.channel.value)
        set_user_id(msg.user_id)

        logger.info(
            "processing_message_started",
            message_id=msg.message_id,
            channel=msg.channel.value,
            has_text=bool(msg.text),
            has_audio=bool(msg.audio_s3_key)
        )

        # 2. Обработка текста (если есть)
        if msg.text:
            # Шаг А: Детекция языка
            detected_lang, confidence = language_detector.detect(msg.text)
            msg.detected_lang = detected_lang
            msg.lang_confidence = confidence

            # Шаг Б: PII-маскирование
            # Если уверенность в языке низкая (< 0.5), используем 'ru' как fallback,
            # так как Presidio лучше всего настроен на русский и кириллицу СНГ.
            target_lang = detected_lang if confidence > 0.5 else "ru"
            msg.masked_text = pii_masker.mask(msg.text, lang=target_lang)

            # Шаг В.0: Получение контекста диалога (для будущего использования в NLLB)
            asyncio.run(context_manager.get_context(msg.user_id, msg.channel))

            # Шаг В: ПЕРЕВОД (Клиент -> Оператор: всегда в русский)
            if detected_lang != "ru" and msg.masked_text:
                logger.info("translating_to_russian", src_lang=detected_lang)
                translated_text = asyncio.run(nllb_translator.translate(
                    text=msg.masked_text,
                    src_lang=detected_lang,
                    tgt_lang="ru"
                ))
                msg.translated_text = translated_text
            else:
                msg.translated_text = msg.masked_text  # Уже на русском

            # Шаг Г: Terminology Override (Пост-обработка переведенного текста)
            final_text = terminology_override.override(msg.translated_text)
            if final_text != msg.translated_text:
                logger.info(
                    "terminology_override_applied",
                    before=msg.translated_text[:50],
                    after=final_text[:50]
                )
            msg.translated_text = final_text

            # Шаг Д: Сохранение реплики клиента в контекст
            client_turn = ConversationTurn(
                role="client",
                original_lang=detected_lang,
                original_text=msg.masked_text or msg.text,
                translated_text=final_text
            )
            asyncio.run(context_manager.add_turn(msg.user_id, msg.channel, client_turn))

            logger.info(
                "text_processing_completed",
                lang=detected_lang,
                conf=round(confidence, 3),
                final_preview=msg.translated_text[:50]
            )
        else:
            msg.translated_text = None
            logger.info("no_text_to_process", message_id=msg.message_id)

        # 6. Интеграция с amoCRM
        try:
            lead_id = asyncio.run(amocrm_client.find_or_create_lead(
                user_id=msg.user_id,
                channel=msg.channel,
                user_display_name=msg.user_display_name
            ))

            operator_note = format_operator_note(msg)
            asyncio.run(amocrm_client.add_note(lead_id, operator_note))

            tags = generate_amo_tags(msg)
            if tags:
                asyncio.run(amocrm_client.update_tags(lead_id, tags))

            logger.info("amocrm_integration_successful", lead_id=lead_id, message_id=msg.message_id, tags=tags)

        except pybreaker.CircuitBreakerError:
            logger.error("amocrm_circuit_open_skipping_crm_update", message_id=msg.message_id)
        except Exception as e:
            logger.error("amocrm_integration_failed", message_id=msg.message_id, error=str(e), exc_info=True)

        logger.info("processing_message_completed", message_id=msg.message_id)
        duration = time.time() - start_time
        translation_latency_histogram.record(duration, {"channel": msg.channel.value, "media_type": "text"})
        if msg.raw_payload.get("fraud_score", 0.0) >= settings.fraud_score_threshold:
            fraud_alerts_counter.add(1, {"channel": msg.channel.value})
        return msg.model_dump()

    except Exception as e:
        logger.error(
            "processing_message_failed",
            message_id=message_dict.get("message_id", "unknown"),
            error=str(e),
            exc_info=True
        )
        raise


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
    start_time = time.time()

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

        # 2. Продвинутое шумоподавление (ЗАМЕНЯЕТ базовый ffmpeg-фильтр из 4.2)
        cleaned_audio_bytes = asyncio.run(audio_preprocessor.reduce_noise(audio_bytes))

        # 3. ASR (Распознавание речи) на очищенном аудио
        asr_result = asyncio.run(whisper_asr.transcribe(cleaned_audio_bytes))

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

            # 5. Получение контекста диалога (для будущего использования в NLLB)
            asyncio.run(context_manager.get_context(msg.user_id, msg.channel))

            # 6. Перевод на русский (если это не русский)
            if detected_lang != "ru":
                translated_text = asyncio.run(nllb_translator.translate(
                    text=masked_text,
                    src_lang=detected_lang,
                    tgt_lang="ru"
                ))
                msg.translated_text = translated_text
            else:
                msg.translated_text = masked_text

            # 7. Terminology Override
            final_text = terminology_override.override(msg.translated_text)
            if final_text != msg.translated_text:
                logger.info(
                    "terminology_override_applied",
                    before=msg.translated_text[:50],
                    after=final_text[:50]
                )
            msg.translated_text = final_text

            # 8. FRAUD DETECTION (Проверяем переведенный текст)
            fraud_score, fraud_flags = fraud_detector.check_text(msg.translated_text)
            msg.raw_payload["fraud_score"] = fraud_score
            msg.raw_payload["fraud_flags"] = fraud_flags

            if fraud_score >= settings.fraud_score_threshold:
                logger.warning(
                    "fraud_alert_triggered",
                    score=fraud_score,
                    flags=fraud_flags,
                    message_id=msg.message_id
                )

            # 9. Сохранение реплики клиента в контекст
            client_turn = ConversationTurn(
                role="client",
                original_lang=detected_lang,
                original_text=masked_text or asr_result.full_text,
                translated_text=final_text
            )
            asyncio.run(context_manager.add_turn(msg.user_id, msg.channel, client_turn))

            # Сохраняем метаданные ASR в raw_payload
            msg.raw_payload["asr_confidence"] = asr_result.avg_confidence
            msg.raw_payload["asr_language"] = asr_result.language

        # 10. Интеграция с amoCRM
        try:
            lead_id = asyncio.run(amocrm_client.find_or_create_lead(
                user_id=msg.user_id,
                channel=msg.channel,
                user_display_name=msg.user_display_name
            ))

            operator_note = format_operator_note(msg)
            asyncio.run(amocrm_client.add_note(lead_id, operator_note))

            tags = generate_amo_tags(msg)
            if tags:
                asyncio.run(amocrm_client.update_tags(lead_id, tags))

            logger.info("amocrm_integration_successful", lead_id=lead_id, message_id=msg.message_id, tags=tags)

        except pybreaker.CircuitBreakerError:
            logger.error("amocrm_circuit_open_skipping_crm_update", message_id=msg.message_id)
        except Exception as e:
            logger.error("amocrm_integration_failed", message_id=msg.message_id, error=str(e), exc_info=True)

        logger.info("voice_processing_completed", message_id=msg.message_id)
        duration = time.time() - start_time
        translation_latency_histogram.record(duration, {"channel": msg.channel.value, "media_type": "voice"})
        if msg.raw_payload.get("fraud_score", 0.0) >= settings.fraud_score_threshold:
            fraud_alerts_counter.add(1, {"channel": msg.channel.value})
        return msg.model_dump()

    except Exception as e:
        logger.error("voice_processing_failed", message_id=message_dict.get("message_id", "unknown"), error=str(e), exc_info=True)
        raise
