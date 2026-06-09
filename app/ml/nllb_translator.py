import asyncio
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.core.config import settings
from app.core.logger import logger

# Маппинг языковых кодов ISO 639-1 в специфичные коды NLLB-200
NLLB_LANG_MAP = {
    "ru": "rus_Cyrl",
    "uz": "uzb_Cyrl",      # Узбекский (кириллица)
    "tg": "tgk_Cyrl",      # Таджикский (кириллица)
    "ky": "kir_Cyrl",      # Киргизский (кириллица)
    "zh": "zho_Hans",      # Китайский (упрощенный)
    "en": "eng_Latn",      # Английский
    "unknown": "rus_Cyrl"  # Fallback на русский
}


class NLLBTranslator:
    def __init__(self):
        self.model_name = settings.nllb_model_name
        self._model: AutoModelForSeq2SeqLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._is_initialized = False
        self._adapter_loaded = False
        logger.info("nllb_translator_initialized_lazy")

    def _initialize(self) -> None:
        """Ленивая инициализация модели и токенизатора."""
        if self._is_initialized:
            return

        logger.info("loading_nllb_model", model=self.model_name, device=settings.nllb_device)

        # 1. Загрузка токенизатора
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 2. Настройка квантования (если включено и есть поддержка)
        load_kwargs = {}
        if settings.nllb_quantize_8bit:
            try:
                import importlib.util
                if importlib.util.find_spec("bitsandbytes"):
                    load_kwargs["load_in_8bit"] = True
                logger.info("nllb_8bit_quantization_enabled")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to full precision")

        # 3. Загрузка модели
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map=settings.nllb_device,
            **load_kwargs
        )

        # Переводим модель в режим оценки (отключает Dropout, экономит память)
        self._model.eval()

        # Отключаем градиенты для инференса
        if not settings.nllb_quantize_8bit:
            self._model = self._model.to(torch.float16 if torch.cuda.is_available() else torch.float32)

        self._is_initialized = True
        logger.info("nllb_model_loaded_successfully")

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Синхронный метод перевода (выполняется в отдельном потоке)."""
        self._initialize()

        if not text or not text.strip():
            return ""

        # Определяем коды языков для NLLB
        src_nllb = NLLB_LANG_MAP.get(src_lang, "rus_Cyrl")
        tgt_nllb = NLLB_LANG_MAP.get(tgt_lang, "rus_Cyrl")

        try:
            # Токенизация с указанием исходного языка
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                src_lang=src_nllb,
                max_length=settings.nllb_max_length,
                truncation=True
            )

            # Перемещаем тензоры на то же устройство, что и модель
            device = self._model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Генерация с ПРИНУДИТЕЛЬНЫМ указанием целевого языка (КРИТИЧЕСКИ ВАЖНО для NLLB)
            forced_bos_token_id = self._tokenizer.lang_code_to_id[tgt_nllb]

            with torch.no_grad(): # Экономия памяти и ускорение
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=settings.nllb_max_length,
                    num_beams=4, # Beam search улучшает качество перевода
                    early_stopping=True
                )

            # Декодирование результата
            translated_text = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            return translated_text.strip()

        except Exception as e:
            logger.error("nllb_translation_failed", text_preview=text[:50], error=str(e), exc_info=True)
            # Fail-soft: возвращаем исходный текст, если модель упала
            return text

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Асинхронная обертка для перевода, не блокирующая event loop."""
        # Используем asyncio.to_thread для выноса блокирующего CPU/GPU вызова в пул потоков
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._translate_sync,
            text,
            src_lang,
            tgt_lang
        )
        return result


    def _check_for_adapter_update(self) -> None:
        """Проверяет, появился ли новый LoRA адаптер, и загружает его."""
        active_adapter_path = "models/nllb_lora_adapter_active"

        if os.path.exists(active_adapter_path) and not self._adapter_loaded:
            logger.info("loading_new_lora_adapter", path=active_adapter_path)
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, active_adapter_path)
                self._adapter_loaded = True
                logger.info("lora_adapter_loaded_successfully")
            except Exception as e:
                logger.error("lora_adapter_load_failed", error=str(e))


# Глобальный синглтон
nllb_translator = NLLBTranslator()
