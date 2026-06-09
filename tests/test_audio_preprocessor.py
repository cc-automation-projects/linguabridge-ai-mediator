from unittest.mock import MagicMock, patch

import pytest

from app.ml.audio_preprocessor import audio_preprocessor


@pytest.mark.asyncio
async def test_denoise_success_mocked():
    fake_audio_bytes = b"fake_ogg_audio_data"

    with patch('librosa.load', return_value=(MagicMock(), 16000)), \
         patch('noisereduce.reduce_noise', return_value=MagicMock()), \
         patch('soundfile.write') as mock_sf_write:

        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = b"cleaned_wav_audio_data"

        with patch('io.BytesIO', return_value=mock_buffer):
            result = await audio_preprocessor.reduce_noise(fake_audio_bytes)

            assert result == b"cleaned_wav_audio_data"
            mock_sf_write.assert_called_once()


@pytest.mark.asyncio
async def test_denoise_fail_soft_on_corrupted_audio():
    corrupted_audio = b"this is not a valid audio file at all"

    with patch('librosa.load', side_effect=Exception("Decoding error")):
        result = await audio_preprocessor.reduce_noise(corrupted_audio)

        assert result == corrupted_audio


@pytest.mark.asyncio
async def test_denoise_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.audio_enable_noise_reduction", False)
    fake_audio = b"original_audio"

    result = await audio_preprocessor.reduce_noise(fake_audio)
    assert result == fake_audio
