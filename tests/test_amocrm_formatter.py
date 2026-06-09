from app.core.models import Channel, IncomingMessage, MediaType
from app.integrations.amocrm_formatter import format_operator_note, generate_amo_tags


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
    assert "⚠️" not in note
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
        raw_payload={"asr_confidence": 0.45}
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
        raw_payload={"fraud_score": 0.95}
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
