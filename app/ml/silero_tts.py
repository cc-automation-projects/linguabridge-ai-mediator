import asyncio
import io

import torch
import torchaudio

from app.core.config import settings
from app.core.logger import logger

SILERO_LANG_MAP = {
    "ru": ("ru", "kseniya"),
    "en": ("en", "bryan"),
    "de": ("de", "thorsten"),
    "es": ("es", "tux"),
    "uk": ("uk", "mykyta"),
}


class SileroTTSService:
    def __init__(self):
        self._models: dict[str, tuple[torch.nn.Module, int]] = {}
        self._lock = asyncio.Lock()
        logger.info("silero_tts_service_initialized_lazy")

    async def _load_model(self, lang_code: str) -> tuple[torch.nn.Module, int]:
        if lang_code in self._models:
            return self._models[lang_code]

        async with self._lock:
            if lang_code in self._models:
                return self._models[lang_code]

            logger.info("loading_silero_model_for_lang", lang=lang_code)
            model, sample_rate = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=lang_code,
                speaker=settings.silero_tts_default_speaker if lang_code == "ru" else None,
                trust_repo=True
            )
            model.eval()
            self._models[lang_code] = (model, sample_rate)
            return self._models[lang_code]

    def _synthesize_sync(self, text: str, lang_code: str) -> bytes:
        model, sample_rate = self._models.get(lang_code)
        if not model:
            raise RuntimeError(f"TTS model for {lang_code} not loaded")

        audio = model.apply_tts(
            text=text,
            speaker=model.speakers[0] if hasattr(model, 'speakers') else settings.silero_tts_default_speaker,
            sample_rate=sample_rate
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0), sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()

    async def synthesize(self, text: str, target_lang: str) -> bytes:
        if not text or not text.strip():
            return b""

        supported_lang = SILERO_LANG_MAP.get(target_lang, ("ru", settings.silero_tts_default_speaker))[0]

        if supported_lang not in self._models:
            await self._load_model(supported_lang)

        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self._synthesize_sync, text, supported_lang)
        return wav_bytes


silero_tts = SileroTTSService()
