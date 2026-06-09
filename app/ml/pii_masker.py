import re

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core.logger import logger


class PIIMaskingService:
    def __init__(self):
        logger.info("pii_masking_service_initializing")

        # 1. Инициализация движков
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # 2. Настройка реестра с кастомными паттернами для РФ
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers() # Загружаем стандартные (PHONE_NUMBER, EMAIL и т.д.)

        # Российский паспорт (формат: 1234 567890 или 1234567890)
        passport_pattern = PatternRecognizer(
            supported_entity="RU_PASSPORT",
            patterns=[re.compile(r'\b\d{4}\s?\d{6}\b')],
            context=["паспорт", "серия", "номер"]
        )

        # Миграционная карта (упрощенно: 10-12 цифр, часто встречается в контексте)
        migration_pattern = PatternRecognizer(
            supported_entity="RU_MIGRATION_CARD",
            patterns=[re.compile(r'\b\d{10,12}\b')],
            context=["миграцион", "карт", "мк"]
        )

        # Российский телефон (форматы: +7(900)123-45-67, 8 900 123 45 67, 89001234567)
        phone_pattern = PatternRecognizer(
            supported_entity="RU_PHONE",
            patterns=[re.compile(r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}')],
            context=["тел", "телефон", "звон"]
        )

        registry.add_recognizer(passport_pattern)
        registry.add_recognizer(migration_pattern)
        registry.add_recognizer(phone_pattern)

        self.analyzer.registry = registry

        # 3. Настройка операторов анонимизации (чем заменять)
        self.operators = {
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[ТЕЛЕФОН_СКРЫТ]"}),
            "RU_PHONE": OperatorConfig("replace", {"new_value": "[ТЕЛЕФОН_СКРЫТ]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_СКРЫТ]"}),
            "RU_PASSPORT": OperatorConfig("replace", {"new_value": "[ПАСПОРТ_СКРЫТ]"}),
            "RU_MIGRATION_CARD": OperatorConfig("replace", {"new_value": "[МИГР_КАРТА_СКРЫТА]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[ИМЯ_СКРЫТО]"}), # Работает благодаря spaCy ru_core_news_sm
            "DEFAULT": OperatorConfig("replace", {"new_value": "[ДАННЫЕ_СКРЫТЫ]"})
        }
        logger.info("pii_masking_service_initialized_with_ru_patterns")

    def mask(self, text: str, lang: str = "ru") -> str:
        """
        Анализирует текст и заменяет PII-сущности на токены.
        """
        if not text or len(text.strip()) < 3:
            return text

        try:
            # Анализ текста. language критически важен для spaCy и контекстных распознавателей
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=["PHONE_NUMBER", "RU_PHONE", "EMAIL_ADDRESS", "RU_PASSPORT", "RU_MIGRATION_CARD", "PERSON"],
                language=lang,
                allow_list=[] # Можно добавить список разрешенных слов, если нужно
            )

            if not analyzer_results:
                return text

            # Анонимизация
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self.operators
            )

            return anonymized_result.text

        except Exception as e:
            # FAIL-SOFT: В продакшене лучше вернуть оригинал и залогировать, чем уронить весь Celery-воркер
            logger.error("pii_masking_failed", text_preview=text[:50], error=str(e))
            return text


# Глобальный синглтон
pii_masker = PIIMaskingService()
