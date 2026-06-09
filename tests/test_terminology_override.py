from app.ml.terminology_override import terminology_override


class TestTerminologyOverride:
    def test_exact_match_replacement(self):
        text = "Мне нужна новая прописка и миграционка."
        result = terminology_override.override(text)
        assert "регистрация по месту жительства" in result
        assert "миграционная карта" in result
        assert "прописка" not in result
        assert "миграционка" not in result

    def test_case_insensitive_match(self):
        text = "Где оформить РВП и СНИЛС?"
        result = terminology_override.override(text)
        # Проверяем, что термины заменены на корректные из словаря
        assert "разрешение на временное проживание (РВП)" in result
        assert "СНИЛС" in result

    def test_no_match_passthrough(self):
        text = "Здравствуйте, как дела? Хочу узнать баланс."
        result = terminology_override.override(text)
        assert result == text

    def test_multiple_overrides_in_sentence(self):
        text = "Для рвп нужен полис омс и дактилоскопия."
        result = terminology_override.override(text)
        assert "разрешение на временное проживание (РВП)" in result
        assert "полис обязательного медицинского страхования (ОМС)" in result
        assert "государственная дактилоскопическая регистрация" in result

    def test_empty_text_handling(self):
        assert terminology_override.override("") == ""
        assert terminology_override.override("   ") == "   "
