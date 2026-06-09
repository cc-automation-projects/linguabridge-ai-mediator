import io

import librosa
import noisereduce as nr
import soundfile as sf

from app.core.config import settings
from app.core.logger import logger


class AdvancedAudioPreprocessor:
    def __init__(self):
        logger.info("advanced_audio_preprocessor_initialized")

    def _reduce_noise_sync(self, audio_bytes: bytes) -> bytes:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

            y_denoised = nr.reduce_noise(
                y=y,
                sr=sr,
                prop_decrease=0.75,
                n_fft=2048,
                stationary=False,
                n_jobs=1
            )

            output_buffer = io.BytesIO()
            sf.write(output_buffer, y_denoised, sr, format='WAV')

            cleaned_bytes = output_buffer.getvalue()
            logger.debug(
                "audio_denoised_successfully",
                original_size=len(audio_bytes),
                cleaned_size=len(cleaned_bytes)
            )
            return cleaned_bytes

        except Exception as e:
            logger.warning(
                "audio_denoising_failed_fallback_to_original",
                error=str(e),
                exc_info=True
            )
            return audio_bytes

    async def reduce_noise(self, audio_bytes: bytes) -> bytes:
        if not settings.audio_enable_noise_reduction:
            return audio_bytes

        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._reduce_noise_sync, audio_bytes)


audio_preprocessor = AdvancedAudioPreprocessor()
