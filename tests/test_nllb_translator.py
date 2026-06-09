
import pytest

from app.ml.nllb_translator import nllb_translator


@pytest.mark.asyncio
async def test_translate_uzbek_to_russian():
    """Проверка перевода с узбекского (кириллица) на русский."""
    text = "Салом, мен патентимни узайтирмоқчиман."
    result = await nllb_translator.translate(text, src_lang="uz", tgt_lang="ru")

    assert isinstance(result, str)
    assert len(result) > 0
    # Проверяем наличие ключевых русских слов (могут быть вариации, но смысл должен быть)
    assert any(word in result.lower() for word in ["привет", "здравств", "патент", "продлить"])


@pytest.mark.asyncio
async def test_translate_tajik_to_russian():
    """Проверка перевода с таджикского на русский."""
    text = "Салом, чӣ хел шумо? Мехоҳам ҳуҷҷатҳоямро дароз кунам."
    result = await nllb_translator.translate(text, src_lang="tg", tgt_lang="ru")

    assert isinstance(result, str)
    assert "документ" in result.lower() or "продлить" in result.lower()


@pytest.mark.asyncio
async def test_translate_russian_to_uzbek():
    """Проверка обратного перевода (для ответов оператора)."""
    text = "Здравствуйте, ваш патент успешно продлен."
    result = await nllb_translator.translate(text, src_lang="ru", tgt_lang="uz")

    assert isinstance(result, str)
    # NLLB должен вывести текст на узбекском
    assert result != text


@pytest.mark.asyncio
async def test_empty_text_handling():
    """Проверка обработки пустого текста."""
    result = await nllb_translator.translate("", src_lang="uz", tgt_lang="ru")
    assert result == ""
