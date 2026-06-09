import os
import shutil
import subprocess
from datetime import datetime

from app.core.celery_app import celery_app
from app.core.logger import logger
from app.ml.fine_tuning.data_extractor import extractor


@celery_app.task(
    name="app.workers.fine_tuning_tasks.run_weekly_finetuning_pipeline",
    queue="ml_training"
)
def run_weekly_finetuning_pipeline():
    logger.info("weekly_finetuning_pipeline_started")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_path = f"data/training_dataset_{timestamp}.jsonl"
    new_adapter_dir = f"models/nllb_lora_adapter_{timestamp}"

    try:
        pairs = extractor.extract_corrections(days_back=7)
        if len(pairs) < 50:
            logger.warning("insufficient_data_for_finetuning", count=len(pairs))
            return "Skipped: Not enough correction data"

        extractor.save_to_jsonl(pairs, dataset_path)

        logger.info("starting_lora_training_script")
        result = subprocess.run(
            ["python", "scripts/train_nllb_lora.py"],
            env={**os.environ, "DATASET_PATH": dataset_path, "OUTPUT_DIR": new_adapter_dir},
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("lora_training_completed", stdout=result.stdout)

        active_adapter_dir = "models/nllb_lora_adapter_active"
        if os.path.exists(active_adapter_dir):
            shutil.move(active_adapter_dir, f"models/nllb_lora_adapter_backup_{timestamp}")

        os.symlink(new_adapter_dir, active_adapter_dir)

        logger.info("canary_deployment_successful", new_version=new_adapter_dir)
        return "Success"

    except subprocess.CalledProcessError as e:
        logger.error("training_script_failed", stderr=e.stderr)
        raise
    except Exception as e:
        logger.error("finetuning_pipeline_failed", error=str(e), exc_info=True)
        raise
