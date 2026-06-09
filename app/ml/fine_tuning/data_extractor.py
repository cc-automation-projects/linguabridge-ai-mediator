import json
import re
from datetime import datetime

from pydantic import BaseModel

from app.core.logger import logger


class TrainingPair(BaseModel):
    src_lang: str
    src_text: str
    tgt_lang: str
    tgt_text: str
    channel: str
    timestamp: str


class DataExtractor:
    def __init__(self):
        self.correction_pattern = re.compile(r"🛠️\s*\*\*Исправлено:\*\*\s*(.+)", re.IGNORECASE | re.DOTALL)
        self.original_pattern = re.compile(r"🔒\s*\*\*Оригинал \(с маскированием PII\):\*\*\s*`(.+?)`", re.IGNORECASE | re.DOTALL)
        self.lang_pattern = re.compile(r"Язык:\s*`([A-Z]{2})`", re.IGNORECASE)
        self.channel_pattern = re.compile(r"Канал:\s*`([A-Z]+)`", re.IGNORECASE)

    async def extract_corrections(self, days_back: int = 7) -> list[TrainingPair]:
        logger.info("starting_data_extraction", days_back=days_back)

        mock_notes = [
            {
                "lead_id": 123,
                "note_text": "🌐 **[AI Медиатор]** Канал: `MAX` | Язык: `UZ`\n📝 **Перевод для оператора:**\nМне нужна регистрация.\n\n---\n🔒 **Оригинал (с маскированием PII):**\n`Menga propiska kerak`\n\n🛠️ **Исправлено:** Мне нужна регистрация по месту жительства."
            }
        ]

        training_pairs = []

        for note in mock_notes:
            text = note["note_text"]

            correction_match = self.correction_pattern.search(text)
            original_match = self.original_pattern.search(text)
            lang_match = self.lang_pattern.search(text)
            channel_match = self.channel_pattern.search(text)

            if correction_match and original_match and lang_match:
                src_text = original_match.group(1).strip()
                tgt_text = correction_match.group(1).strip()
                lang = lang_match.group(1).lower()
                channel = channel_match.group(1).lower() if channel_match else "unknown"

                lang_map = {"uz": "uzb_Cyrl", "tg": "tgk_Cyrl", "ky": "kir_Cyrl", "ru": "rus_Cyrl"}
                src_nllb = lang_map.get(lang, "rus_Cyrl")

                pair = TrainingPair(
                    src_lang=src_nllb,
                    src_text=src_text,
                    tgt_lang="rus_Cyrl",
                    tgt_text=tgt_text,
                    channel=channel,
                    timestamp=datetime.utcnow().isoformat()
                )
                training_pairs.append(pair)

        logger.info("data_extraction_completed", pairs_count=len(training_pairs))
        return training_pairs

    async def save_to_jsonl(self, pairs: list[TrainingPair], output_path: str = "data/training_dataset.jsonl"):
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_jsonl, pairs, output_path)

    def _write_jsonl(self, pairs: list[TrainingPair], output_path: str):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair.model_dump(), ensure_ascii=False) + '\n')
        logger.info("dataset_saved_to_jsonl", path=output_path, records=len(pairs))


extractor = DataExtractor()
