from app.core.config import settings
from app.core.models import IncomingMessage


def format_operator_note(msg: IncomingMessage) -> str:
    channel_name = msg.channel.value.upper()
    lang_name = msg.detected_lang.upper() if msg.detected_lang else "UNKNOWN"

    note = f"🌐 **[AI Медиатор]** Канал: `{channel_name}` | Язык: `{lang_name}`\n"

    alerts = []

    fraud_score = msg.raw_payload.get("fraud_score", 0.0)
    if fraud_score >= settings.fraud_score_threshold:
        alerts.append("🚨 **ВНИМАНИЕ: Высокая вероятность мошенничества!** Проверьте запрос перед действием.")

    if msg.media_type:
        asr_conf = msg.raw_payload.get("asr_confidence", 1.0)
        if asr_conf < settings.min_asr_confidence:
            alerts.append(f"⚠️ **Низкая уверенность распознавания речи ({asr_conf:.2f}).** Текст может содержать ошибки. Рекомендуется уточнить у клиента.")

    if alerts:
        note += "\n" + "\n".join(alerts) + "\n"

    note += "\n"

    note += f"📝 **Перевод для оператора:**\n{msg.translated_text or '[Текст отсутствует]'}\n\n"

    note += "---\n"
    note += f"🔒 **Оригинал (с маскированием PII):**\n`{msg.masked_text or msg.text or '[Пусто]'}`\n"

    if msg.audio_s3_key:
        note += f"\n📎 **Аудиофайл:** `{msg.audio_s3_key}`\n"

    return note.strip()


def generate_amo_tags(msg: IncomingMessage) -> list[str]:
    tags = []

    tags.append(f"channel_{msg.channel.value}")

    if msg.detected_lang and msg.detected_lang != "unknown":
        tags.append(f"lang_{msg.detected_lang}")

    if msg.media_type:
        tags.append(f"media_{msg.media_type.value}")

    if msg.raw_payload.get("fraud_score", 0.0) >= settings.fraud_score_threshold:
        tags.append("⚠️_fraud_alert")

    return list(set(tags))
