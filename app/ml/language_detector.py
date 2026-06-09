import os

import fasttext

from app.core.logger import logger


class LanguageDetector:
    def __init__(self, model_path: str = "app/ml/lid.176.bin"):
        self.model_path = model_path
        self._model = None
        logger.info("language_detector_initialized_lazy")

    @property
    def model(self):
        """Lazy loading модели для экономии памяти при старте."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"FastText model not found at {self.model_path}. "
                    "Please download it: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O app/ml/lid.176.bin"
                )
            # suppress_output подавляет спам в консоль при загрузке
            self._model = fasttext.load_model(self.model_path)
            logger.info("fasttext_model_loaded_successfully", path=self.model_path)
        return self._model

    def detect(self, text: str) -> tuple[str, float]:
        """
        Определяет язык текста.
        :return: Кортеж (код языка ISO 639-1, уверенность от 0.0 до 1.0)
        """
        if not text or len(text.strip()) < 2:
            return "unknown", 0.0

        try:
            # FastText чувствителен к переносам строк, заменяем их на пробелы
            clean_text = text.replace('\n', ' ').replace('\r', ' ')

            # k=1 возвращает только лучший результат
            predictions = self.model.predict(clean_text, k=1)

            # Формат вывода: ('__label__ru',) и (0.9876,)
            lang_code = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])

            return lang_code, confidence

        except Exception as e:
            logger.error("language_detection_failed", text_preview=text[:30], error=str(e))
            # Fail-soft: возвращаем 'ru' как дефолт для РФ-контекста, но с низкой уверенностью
            return "ru", 0.0


# Глобальный синглтон
language_detector = LanguageDetector()
