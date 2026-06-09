from app.core.config import settings
from app.ml.audio_utils import reduce_audio_noise
from app.ml.fraud_detector import fraud_detector


class TestFraudDetector:
    def test_no_fraud_detected(self):
        text = "Здравствуйте, хочу узнать график работы офиса."
        score, flags = fraud_detector.check_text(text)
        assert score == 0.0
        assert len(flags) == 0

    def test_single_fraud_pattern(self):
        text = "Мне нужно перевести деньги на безопасный счет."
        score, flags = fraud_detector.check_text(text)
        assert score >= 0.8
        assert "безопасный счет" in flags
        assert "переведите деньги" in flags

    def test_multiple_fraud_patterns_aggregation(self):
        text = "Это сотрудник полиции. Назовите код из смс и никому не говорите."
        score, flags = fraud_detector.check_text(text)
        assert score == 1.0
        assert len(flags) >= 2

    def test_case_insensitive(self):
        text = "Сотрудник Полиции просит удалить приложение."
        score, flags = fraud_detector.check_text(text)
        assert score > 0.7
        assert "сотрудник полиции" in flags


class TestAudioUtils:
    def test_noise_reduction_passthrough_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "audio_enable_noise_reduction", False)
        dummy_audio = b"fake_audio_data"
        result = reduce_audio_noise(dummy_audio)
        assert result == dummy_audio

    def test_noise_reduction_fail_soft_on_invalid_audio(self):
        invalid_audio = b"this is not an ogg file"
        result = reduce_audio_noise(invalid_audio)
        assert result == invalid_audio
