import asyncio
import uuid

from app.adapters.max_adapter import max_client
from app.adapters.telegram_adapter import telegram_client
from app.adapters.vk_adapter import vk_client
from app.core.celery_app import celery_app
from app.core.logger import logger
from app.core.models import Channel
from app.infrastructure.audio_converter import audio_converter
from app.infrastructure.s3 import s3_service
from app.ml.nllb_translator import nllb_translator
from app.ml.silero_tts import silero_tts


@celery_app.task(
    bind=True,
    name="app.workers.tts_tasks.generate_and_send_voice_reply",
    queue="tts_generation",
    acks_late=True,
    reject_on_worker_lost=True,
    time_limit=120
)
def generate_and_send_voice_reply(self, task_data: dict) -> str:
    chat_id = task_data["chat_id"]
    channel = Channel(task_data["channel"])
    target_lang = task_data["target_lang"]
    operator_text = task_data["operator_text_ru"]

    trace_id = str(uuid.uuid4())
    logger.info("voice_reply_pipeline_started", trace_id=trace_id, channel=channel.value)

    try:
        if target_lang != "ru":
            translated_text = asyncio.run(nllb_translator.translate(
                text=operator_text,
                src_lang="ru",
                tgt_lang=target_lang
            ))
        else:
            translated_text = operator_text

        logger.info("operator_text_translated", target_lang=target_lang, preview=translated_text[:50])

        wav_bytes = asyncio.run(silero_tts.synthesize(translated_text, target_lang))

        ogg_bytes = audio_converter.convert_wav_to_ogg_opus(wav_bytes, target_sr=24000)

        file_key = f"{channel.value}/tts_reply/{uuid.uuid4()}.ogg"
        success = asyncio.run(s3_service.upload_file(file_key, ogg_bytes, content_type="audio/ogg"))
        if not success:
            raise RuntimeError("Failed to upload TTS audio to S3")

        logger.info("tts_audio_uploaded_to_s3", file_key=file_key)

        if channel == Channel.MAX:
            asyncio.run(max_client.send_voice(chat_id, file_key))
        elif channel == Channel.VK:
            asyncio.run(vk_client.send_voice(chat_id, file_key))
        elif channel == Channel.TELEGRAM:
            asyncio.run(telegram_client.send_voice(chat_id, ogg_bytes))

        logger.info("voice_reply_sent_successfully", chat_id=chat_id, trace_id=trace_id)
        return file_key

    except Exception as e:
        logger.error("voice_reply_pipeline_failed", trace_id=trace_id, error=str(e), exc_info=True)
        raise
