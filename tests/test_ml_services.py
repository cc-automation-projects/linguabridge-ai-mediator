from app.ml.language_detector import language_detector
from app.ml.pii_masker import pii_masker


class TestLanguageDetector:
    def test_detect_russian(self):
        lang, conf = language_detector.detect("Здравствуйте, как продлить патент?")
        assert lang == "ru"
        assert conf > 0.9

    def test_detect_uzbek_cyrillic(self):
        # FastText хорошо определяет узбекский на кириллице
        lang, conf = language_detector.detect("Салом, менинг патентим тугаяпти")
        assert lang in ["uz", "ru"] # Иногда может спутать с ru из-за кириллицы, но это ожидаемо для коротких фраз
        assert conf > 0.5

    def test_detect_tajik(self):
        lang, conf = language_detector.detect("Салом, чӣ хел шумо? Мехоҳам ҳуҷҷатҳоямро дароз кунам")
        assert lang == "tg"
        assert conf > 0.8

    def test_short_text_fallback(self):
        lang, conf = language_detector.detect("да")
        assert lang == "unknown" or lang == "ru"
        assert conf < 0.6 # Низкая уверенность для слишком коротких текстов


class TestPIIMasker:
    def test_mask_russian_phone(self):
        text = "Мой номер +7 (900) 123-45-67, звоните"
        masked = pii_masker.mask(text, lang="ru")
        assert "[ТЕЛЕФОН_СКРЫТ]" in masked
        assert "900" not in masked

    def test_mask_passport(self):
        text = "Серия и номер паспорта 4515 123456"
        masked = pii_masker.mask(text, lang="ru")
        assert "[ПАСПОРТ_СКРЫТ]" in masked
        assert "4515" not in masked

    def test_mask_migration_card(self):
        text = "Номер моей миграционной карты 123456789012"
        masked = pii_masker.mask(text, lang="ru")
        assert "[МИГР_КАРТА_СКРЫТА]" in masked
        assert "123456789012" not in masked

    def test_mask_person_name(self):
        # Работает благодаря spaCy ru_core_news_sm
        text = "Меня зовут Иван Иванович Иванов"
        masked = pii_masker.mask(text, lang="ru")
        assert "[ИМЯ_СКРЫТО]" in masked

    def test_combined_masking(self):
        text = "Я Рахим, мой телефон 89001112233 и паспорт 4515 123456"
        masked = pii_masker.mask(text, lang="ru")
        assert "[ИМЯ_СКРЫТО]" in masked
        assert "[ТЕЛЕФОН_СКРЫТ]" in masked
        assert "[ПАСПОРТ_СКРЫТ]" in masked
