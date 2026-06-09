import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "facebook/nllb-200-distilled-600M"
DATASET_PATH = "data/training_dataset.jsonl"
OUTPUT_DIR = "models/nllb_lora_adapter_v1"
SRC_LANG = "uzb_Cyrl"
TGT_LANG = "rus_Cyrl"


def main():
    print("🚀 Запуск Fine-Tuning NLLB-200 с LoRA...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATASET_PATH)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["src_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            text_target=examples["tgt_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

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

    print("🏋️ Начало обучения...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Обучение завершено. Адаптер сохранен в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
