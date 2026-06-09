import ffmpeg

from app.core.config import settings
from app.core.logger import logger


def reduce_audio_noise(audio_bytes: bytes) -> bytes:
    if not settings.audio_enable_noise_reduction:
        return audio_bytes

    try:
        input_stream = ffmpeg.input('pipe:0', format='ogg')

        filtered_stream = (
            input_stream
            .filter('highpass', f=settings.audio_highpass_freq)
            .filter('lowpass', f=settings.audio_lowpass_freq)
            .filter('afftdn', nf=-15)
            .output('pipe:1', format='ogg', acodec='libopus')
        )

        out_bytes, _ = filtered_stream.run(input=audio_bytes, capture_stdout=True, capture_stderr=True)

        logger.debug("audio_noise_reduction_applied", original_size=len(audio_bytes), filtered_size=len(out_bytes))
        return out_bytes

    except ffmpeg.Error as e:
        logger.warning("audio_filtering_failed_fallback_to_original", stderr=e.stderr.decode('utf-8', errors='ignore'))
        return audio_bytes
    except Exception as e:
        logger.error("audio_processing_unexpected_error", error=str(e))
        return audio_bytes
