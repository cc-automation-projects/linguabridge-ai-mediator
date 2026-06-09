import json
import os

import ahocorasick

from app.core.logger import logger


class TerminologyOverrideService:
    def __init__(self, dict_path: str = "app/ml/terminology_dict.json"):
        self.dict_path = dict_path
        self._automaton = None
        self._terms_map: dict[str, str] = {}
        self._is_initialized = False
        logger.info("terminology_override_service_initialized_lazy")

    def _initialize(self) -> None:
        """Ленивая загрузка словаря и построение автомата Aho-Corasick."""
        if self._is_initialized:
            return

        if not os.path.exists(self.dict_path):
            logger.warning("terminology_dict_not_found", path=self.dict_path)
            self._is_initialized = True
            return

        try:
            with open(self.dict_path, encoding='utf-8') as f:
                self._terms_map = json.load(f)

            self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY)

            # Добавляем все ключи в автомат в нижнем регистре для нечувствительного к регистру поиска
            for key, value in self._terms_map.items():
                self._automaton.add_word(key.lower(), (key.lower(), value))

            self._automaton.make_automaton()
            self._is_initialized = True
            logger.info("terminology_automaton_built_successfully", terms_count=len(self._terms_map))

        except Exception as e:
            logger.error("terminology_override_init_failed", error=str(e), exc_info=True)
            self._is_initialized = True # Чтобы не пытаться снова при каждом вызове

    def override(self, text: str) -> str:
        """
        Заменяет неточные термины на корректные в заданном тексте.
        Работает за O(N), где N - длина текста.
        """
        if not text or not text.strip():
            return text

        self._initialize()

        if not self._automaton or not self._terms_map:
            return text

        text_lower = text.lower()

        # Находим все совпадения.
        # automaton.iter возвращает генератор: (end_index, (matched_key, replacement_value))
        matches = list(self._automaton.iter(text_lower))

        if not matches:
            return text

        # Сортируем совпадения по конечному индексу в ОБРАТНОМ порядке (справа налево).
        # Это критически важно: при замене строки её длина может измениться,
        # и индексы следующих совпаданий сдвинутся. Замена справа налево это предотвращает.
        matches.sort(key=lambda x: x[0], reverse=True)

        result = text
        for end_index, (matched_key, replacement) in matches:
            start_index = end_index - len(matched_key) + 1

            # Опционально: можно добавить логику сохранения исходного регистра первого символа,
            # но для юридических терминов лучше использовать строго заданный в словаре регистр.
            result = result[:start_index] + replacement + result[end_index + 1:]

        return result


# Глобальный синглтон
terminology_override = TerminologyOverrideService()
