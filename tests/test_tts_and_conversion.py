
import pytest

from app.infrastructure.audio_converter import audio_converter


def test_wav_to_ogg_conversion():
    dummy_wav = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xBB\x00\x00\x00\xEE\x02\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    ogg_bytes = audio_converter.convert_wav_to_ogg_opus(dummy_wav)
    assert len(ogg_bytes) > 0
    assert ogg_bytes[:4] == b"OggS"


@pytest.mark.asyncio
async def test_silero_tts_initialization():
    import torch
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        pytest.skip("MPS backend not fully supported for this test")
    pass
