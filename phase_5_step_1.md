Здесь мы превращаем систему из "просто работающей" в "постоянно улучшающуюся". Базовая модель NLLB-200 хороша, но она не знает вашей специфики, сленга мигрантов и внутренних терминов компании. Мы реализуем петлю обратной связи (Feedback Loop), где исправления операторов автоматически превращаются в обучающие данные для дообучения модели.

---

# ЭТАП 5, ПОДЗАДАЧА 5.1: Пайплайн LoRA Fine-Tuning для NLLB

## Шаг 5.1.1: Зависимости и системные требования

Для эффективного дообучения (Fine-Tuning) нам понадобятся специализированные библиотеки. **Важно:** Этот пайплайн требует наличия GPU (минимум 1x NVIDIA T4/A10G с 16GB VRAM) на выделенном воркере или сервере.

**1. Обновите `pyproject.toml`:**
```toml
# === Fine-Tuning & MLOps ===
peft = "^0.9.0"
datasets = "^2.18.0"
accelerate = "^0.27.2"
wandb = "^0.16.0" # Опционально, для трекинга метрик обучения
```
*Действие:* Выполните `poetry install`.

---

## Шаг 5.1.2: Скрипт извлечения "Золотого датасета" из amoCRM

Нам нужен механизм, который находит случаи, когда оператор **вручную исправил** перевод AI. 
*Договоренность с бизнесом:* Операторы должны использовать специальный тег или формат в примечании, если они правят перевод. Например: `🛠️ **Исправлено:** [правильный текст]`.

**Файл: `app/ml/fine_tuning/data_extractor.py`**
```python
import re
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pydantic import BaseModel

from app.integrations.amocrm_client import amocrm_client
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
        # Регулярное выражение для поиска исправлений операторов
        # Пример совпадения: "🛠️ **Исправлено:** клиент хочет продлить патент"
        self.correction_pattern = re.compile(r"🛠️\s*\*\*Исправлено:\*\*\s*(.+)", re.IGNORECASE | re.DOTALL)
        self.original_pattern = re.compile(r"🔒\s*\*\*Оригинал \(с маскированием PII\):\*\*\s*`(.+?)`", re.IGNORECASE | re.DOTALL)
        self.lang_pattern = re.compile(r"Язык:\s*`([A-Z]{2})`", re.IGNORECASE)
        self.channel_pattern = re.compile(r"Канал:\s*`([A-Z]+)`", re.IGNORECASE)

    async def extract_corrections(self, days_back: int = 7) -> List[TrainingPair]:
        """
        Извлекает пары "оригинал -> исправленный перевод" из amoCRM за последние N дней.
        """
        logger.info(f"starting_data_extraction", days_back=days_back)
        
        # В реальном сценарии здесь будет запрос к API amoCRM с фильтром по дате и наличию тега "ai_corrected"
        # Для примера эмулируем получение списка лидов с нужными примечаниями
        # Примечание: В продакшене нужно использовать метод amocrm_client.get_leads_with_corrections()
        
        mock_notes = [
            {
                "lead_id": 123,
                "note_text": "🌐 **[AI Медиатор]** Канал: `MAX` | Язык: `UZ`\n📝 **Перевод для оператора:**\nМне нужна регистрация.\n\n---\n🔒 **Оригинал (с маскированием PII):**\n`Menga propiska kerak`\n\n🛠️ **Исправлено:** Мне нужна регистрация по месту жительства."
            }
        ]
        
        training_pairs = []
        
        for note in mock_notes: # Заменить на реальный асинхронный вызов к amoCRM
            text = note["note_text"]
            
            # Ищем исправление
            correction_match = self.correction_pattern.search(text)
            original_match = self.original_pattern.search(text)
            lang_match = self.lang_pattern.search(text)
            channel_match = self.channel_pattern.search(text)
            
            if correction_match and original_match and lang_match:
                src_text = original_match.group(1).strip()
                tgt_text = correction_match.group(1).strip()
                lang = lang_match.group(1).lower()
                channel = channel_match.group(1).lower() if channel_match else "unknown"
                
                # Маппинг кодов языков в формат NLLB
                lang_map = {"uz": "uzb_Cyrl", "tg": "tgk_Cyrl", "ky": "kir_Cyrl", "ru": "rus_Cyrl"}
                src_nllb = lang_map.get(lang, "rus_Cyrl")
                
                pair = TrainingPair(
                    src_lang=src_nllb,
                    src_text=src_text,
                    tgt_lang="rus_Cyrl", # Мы учим переводить НА русский (или наоборот, если нужно)
                    tgt_text=tgt_text,
                    channel=channel,
                    timestamp=datetime.utcnow().isoformat()
                )
                training_pairs.append(pair)
                
        logger.info("data_extraction_completed", pairs_count=len(training_pairs))
        return training_pairs

    async def save_to_jsonl(self, pairs: List[TrainingPair], output_path: str = "data/training_dataset.jsonl"):
        """Сохраняет пары в формат JSONL, ожидаемый Hugging Face Datasets."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair.model_dump(), ensure_ascii=False) + '\n')
                
        logger.info("dataset_saved_to_jsonl", path=output_path, records=len(pairs))

extractor = DataExtractor()
```

---

## Шаг 5.1.3: Скрипт LoRA Fine-Tuning (Ядро MLOps)

Это автономный скрипт, который запускается на GPU-машине. Он загружает базовую модель, применяет адаптеры LoRA и дообучает её на нашем JSONL-датасете.

**Файл: `scripts/train_nllb_lora.py`**
```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# === Конфигурация ===
MODEL_ID = "facebook/nllb-200-distilled-600M"
DATASET_PATH = "data/training_dataset.jsonl"
OUTPUT_DIR = "models/nllb_lora_adapter_v1"
SRC_LANG = "uzb_Cyrl"  # Можно сделать динамическим, если обучаем на миксе языков
TGT_LANG = "rus_Cyrl"

def main():
    print("🚀 Запуск Fine-Tuning NLLB-200 с LoRA...")

    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        load_in_8bit=True, # Экономия VRAM
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 2. Настройка LoRA
    # Целевые модули для NLLB (трансформерные слои)
    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, # Ранг матриц LoRA (8-16 оптимально для NLLB)
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # Должно быть ~1-2% обучаемых параметров

    # 3. Загрузка и предобработка датасета
    dataset = load_dataset("json", data_files=DATASET_PATH)

    def preprocess_function(examples):
        # Формируем входные данные с принудительным токеном языка (как в инференсе)
        inputs = [f"{src} </s>" for src in examples["src_text"]] # Упрощенно, в реальности нужен src_lang токен
        
        # Для NLLB правильный формат:
        model_inputs = tokenizer(
            examples["src_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        # Таргет с принудительным BOS токеном целевого языка
        labels = tokenizer(
            text_target=examples["tgt_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # 4. Настройка Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="no",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # 5. Запуск обучения
    print("🏋️ Начало обучения...")
    trainer.train()

    # 6. Сохранение только адаптеров LoRA (весят ~10-20 МБ, а не гигабайты)
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Обучение завершено. Адаптер сохранен в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

---

## Шаг 5.1.4: Автоматизация пайплайна и Canary Deployment

Чтобы это работало "само", мы создаем Celery-задачу, которая запускается по расписанию (например, раз в неделю), собирает данные, запускает обучение (через subprocess или отдельный сервис) и обновляет модель.

**Файл: `app/workers/fine_tuning_tasks.py`**
```python
import os
import subprocess
import shutil
from datetime import datetime
from app.core.celery_app import celery_app
from app.ml.fine_tuning.data_extractor import extractor
from app.core.logger import logger

@celery_app.task(
    name="app.workers.fine_tuning_tasks.run_weekly_finetuning_pipeline",
    queue="ml_training" # Требует выделенного GPU-воркера
)
def run_weekly_finetuning_pipeline():
    """
    Еженедельный пайплайн: Извлечение данных -> Обучение -> Деплой.
    """
    logger.info("weekly_finetuning_pipeline_started")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_path = f"data/training_dataset_{timestamp}.jsonl"
    new_adapter_dir = f"models/nllb_lora_adapter_{timestamp}"

    try:
        # 1. Извлечение данных
        pairs = extractor.extract_corrections(days_back=7)
        if len(pairs) < 50:
            logger.warning("insufficient_data_for_finetuning", count=len(pairs))
            return "Skipped: Not enough correction data"
            
        extractor.save_to_jsonl(pairs, dataset_path)

        # 2. Запуск скрипта обучения (в реальном продакшене лучше использовать Ray или Kubeflow)
        logger.info("starting_lora_training_script")
        result = subprocess.run(
            ["python", "scripts/train_nllb_lora.py"],
            env={**os.environ, "DATASET_PATH": dataset_path, "OUTPUT_DIR": new_adapter_dir},
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("lora_training_completed", stdout=result.stdout)

        # 3. Canary Deployment (Обновление активной версии)
        active_adapter_dir = "models/nllb_lora_adapter_active"
        if os.path.exists(active_adapter_dir):
            # Сохраняем бэкап предыдущей версии
            shutil.move(active_adapter_dir, f"models/nllb_lora_adapter_backup_{timestamp}")
            
        # Создаем симлинк на новую версию (мгновенный "деплой" без перезагрузки Python-процесса, если реализован hot-reload)
        os.symlink(new_adapter_dir, active_adapter_dir)
        
        logger.info("canary_deployment_successful", new_version=new_adapter_dir)
        return "Success"

    except subprocess.CalledProcessError as e:
        logger.error("training_script_failed", stderr=e.stderr)
        raise
    except Exception as e:
        logger.error("finetuning_pipeline_failed", error=str(e), exc_info=True)
        raise
```

---

## Шаг 5.1.5: Интеграция адаптера в основной пайплайн (Hot-Reload)

Чтобы основной сервис (`nllb_translator.py`) начал использовать новый адаптер без перезагрузки всего воркера, добавим логику проверки обновлений.

**Обновите `app/ml/nllb_translator.py` (добавьте метод):**
```python
    def _check_for_adapter_update(self) -> None:
        """Проверяет, появился ли новый LoRA адаптер, и загружает его."""
        active_adapter_path = "models/nllb_lora_adapter_active"
        
        if os.path.exists(active_adapter_path) and not self._adapter_loaded:
            logger.info("loading_new_lora_adapter", path=active_adapter_path)
            try:
                from peft import PeftModel
                # Загружаем адаптер поверх базовой модели
                self._model = PeftModel.from_pretrained(self._model, active_adapter_path)
                self._adapter_loaded = True
                logger.info("lora_adapter_loaded_successfully")
            except Exception as e:
                logger.error("lora_adapter_load_failed", error=str(e))
```
*Этот метод можно вызывать раз в час или при каждом N-ном запросе для проверки актуальности.*

---

## Шаг 5.1.6: Production-нюансы Fine-Tuning

1. **Качество данных > Количество:** 100 пар *качественных* исправлений от опытных операторов ценнее, чем 10 000 пар с ошибками. Внедрите валидацию: если оператор исправил текст, но новый текст все равно содержит PII или бессмыслицу, не добавляйте эту пару в датасет.
2. **Катастрофическая забывчивость (Catastrophic Forgetting):** Дообучая модель на узком домене (например, только миграционные вопросы), она может "забыть", как переводить общие фразы. 
   * *Решение:* Всегда добавляйте в обучающий датасет ~10-20% случайных "хороших" пар из оригинального датасета NLLB (или ваших старых проверенных данных) для регуляризации.
3. **Ресурсы:** Обучение даже с LoRA требует GPU. В облаке (Yandex Cloud, AWS) дешевле всего запускать этот пайплайн на Spot-инстансах с GPU (например, NVIDIA A10G), так как задача не критична к времени выполнения (можно подождать, пока выделится дешевый инстанс).

---

### Что мы достигли в Подзадаче 5.1:

✅ **Замкнутый цикл улучшения (Feedback Loop):** Ошибки AI больше не теряются. Каждое исправление оператора становится ценным активом, делающим систему умнее.  
✅ **Экономичное дообучение:** Использование LoRA позволяет дообучить большую модель (600M параметров), обновляя лишь ~1-2% весов. Это занимает минуты/часы на одном GPU и требует минимум места для хранения адаптеров (~20 МБ).  
✅ **Автоматизация MLOps:** Пайплайн от извлечения данных из amoCRM до деплоя новой версии модели через симлинк (Canary) полностью автоматизирован.  
✅ **Адаптация под домен:** Модель перестает быть "усредненной" и становится экспертом именно в вашей предметной области и специфическом сленге ваших клиентов.  
