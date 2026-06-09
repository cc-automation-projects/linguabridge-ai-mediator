
import ahocorasick

from app.core.logger import logger


class FraudDetectorService:
    def __init__(self):
        self._automaton = None
        self._patterns_weights = {}
        self._is_initialized = False
        logger.info("fraud_detector_service_initialized_lazy")

    def _initialize(self) -> None:
        if self._is_initialized:
            return

        patterns = [
            ("безопасный счет", 0.9),
            ("сотрудник полиции", 0.8),
            ("служба безопасности банка", 0.8),
            ("переведите деньги", 0.7),
            ("код из смс", 0.9),
            ("никому не говорите", 0.8),
            ("решить вопрос с документами", 0.6),
            ("взяли кредит на ваше имя", 0.8),
            ("демонстрация экрана", 0.7),
            ("удалите приложение", 0.9),
        ]

        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY)
        for pattern, weight in patterns:
            self._automaton.add_word(pattern.lower(), (pattern, weight))

        self._automaton.make_automaton()
        self._is_initialized = True
        logger.info("fraud_detector_automaton_built", patterns_count=len(patterns))

    def check_text(self, text: str) -> tuple[float, list[str]]:
        if not text or not text.strip():
            return 0.0, []

        self._initialize()

        text_lower = text.lower()
        matches = list(self._automaton.iter(text_lower))

        if not matches:
            return 0.0, []

        triggered_patterns = set()
        max_weight = 0.0

        for _, (pattern, weight) in matches:
            triggered_patterns.add(pattern)
            if weight > max_weight:
                max_weight = weight

        final_score = min(1.0, max_weight + (len(triggered_patterns) - 1) * 0.1)

        return round(final_score, 2), list(triggered_patterns)


fraud_detector = FraudDetectorService()
