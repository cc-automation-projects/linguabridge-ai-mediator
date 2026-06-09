
import asyncio

import pytest

from app.ml.whisper_asr import whisper_asr


def _load_audio_fixture(path: str) -> bytes:
    import os
    if not os.path.exists(path):
        pytest.skip("Test audio fixture not found. Skipping ASR test.")
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.asyncio
async def test_whisper_transcribe_real_audio():
    """Интеграционный тест распознавания реального аудиофайла."""
    loop = asyncio.get_running_loop()
    audio_bytes = await loop.run_in_executor(None, _load_audio_fixture, "tests/fixtures/test_voice.ogg")

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
