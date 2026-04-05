# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "transformers>=4.44.0",
#     "trl>=0.12.0",
#     "peft>=0.13.0",
#     "bitsandbytes>=0.43.0",
#     "datasets>=3.0.0",
#     "mlflow>=2.16.0",
#     "accelerate>=0.34.0",
# ]
# ///
"""
LoRA SFT training template for Databricks serverless GPUs.

This is the BASE training script that the auto-research agent modifies
per experiment. The CONFIG dict at the top is the injection point.

Usage:
    Submitted as a Databricks job via databricks-sdk. The orchestrator
    modifies CONFIG and submits the script. The final stdout line is
    `val_loss: {value}` for the orchestrator to parse.
"""

import json
import os
import sys
import time
import hashlib

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ──────────────────────────────────────────────────────────────
# CONFIG — the auto-research agent modifies this dict per run
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "base_model": "Qwen/Qwen2.5-3B",
    "dataset_path": "/Volumes/catalog/schema/autoresearch/data/train.jsonl",
    "val_dataset_path": "/Volumes/catalog/schema/autoresearch/data/val.jsonl",
    "adapter_save_path": "/Volumes/catalog/schema/autoresearch/adapters",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "max_steps": 200,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_length": 1024,
    "use_qlora": True,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "eval_steps": 50,
    "experiment_name": None,  # set by orchestrator, e.g. "/Users/user/autoresearch/exp1"
    "run_name": None,         # set by orchestrator
    "experiment_description": "",
    "auto_research_iteration": 0,
}

# Allow CONFIG override via CLI argument (JSON string)
if len(sys.argv) > 1:
    override = json.loads(sys.argv[1])
    CONFIG.update(override)


def compute_dataset_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    # ── MLflow setup ──────────────────────────────────────────
    mlflow.set_tracking_uri("databricks")
    if CONFIG["experiment_name"]:
        mlflow.set_experiment(CONFIG["experiment_name"])

    # ── Quantization config (QLoRA) ──────────────────────────
    bnb_config = None
    torch_dtype = "auto"
    if CONFIG["use_qlora"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.bfloat16

    # ── Load model ───────────────────────────────────────────
    print(f"Loading model: {CONFIG['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Load dataset ─────────────────────────────────────────
    print(f"Loading dataset: {CONFIG['dataset_path']}")
    data_files = {"train": CONFIG["dataset_path"]}
    if CONFIG["val_dataset_path"] and os.path.exists(CONFIG["val_dataset_path"]):
        data_files["validation"] = CONFIG["val_dataset_path"]

    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    dataset_hash = compute_dataset_hash(CONFIG["dataset_path"])
    print(f"Dataset hash: {dataset_hash}, train size: {len(train_dataset)}")

    # ── Training config ──────────────────────────────────────
    run_name = CONFIG["run_name"] or f"autoresearch-run-{int(time.time())}"
    training_config = SFTConfig(
        output_dir="/tmp/sft_output",
        max_steps=CONFIG["max_steps"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=CONFIG["warmup_steps"],
        bf16=True,
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="no",
        max_length=CONFIG["max_length"],
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        report_to="mlflow",
        run_name=run_name,
        dataloader_num_workers=2,
        optim="paged_adamw_8bit" if CONFIG["use_qlora"] else "adamw_torch",
    )

    # ── Train ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    # ── Evaluate ─────────────────────────────────────────────
    val_loss = float("inf")
    if eval_dataset:
        eval_result = trainer.evaluate()
        val_loss = eval_result["eval_loss"]

    # ── Log custom metrics and artifacts ─────────────────────
    vram_peak_gb = 0.0
    if torch.cuda.is_available():
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    mlflow.log_metrics({
        "val_loss": val_loss,
        "vram_peak_gb": round(vram_peak_gb, 2),
        "wall_time_seconds": round(wall_time, 1),
    })

    mlflow.set_tags({
        "experiment_description": CONFIG["experiment_description"],
        "keep_or_revert": "pending",
        "base_model": CONFIG["base_model"],
        "dataset_hash": dataset_hash,
        "training_method": "qlora_sft" if CONFIG["use_qlora"] else "lora_sft",
        "auto_research_iteration": str(CONFIG["auto_research_iteration"]),
    })

    # Log training config as artifact
    config_path = "/tmp/training_config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)
    mlflow.log_artifact(config_path, artifact_path="config")

    # Log this script as artifact
    mlflow.log_artifact(os.path.abspath(__file__), artifact_path="scripts")

    # ── Save LoRA adapter ────────────────────────────────────
    run_id = mlflow.active_run().info.run_id
    adapter_path = os.path.join(CONFIG["adapter_save_path"], run_id)
    os.makedirs(adapter_path, exist_ok=True)
    trainer.save_model(adapter_path)
    mlflow.log_artifacts(adapter_path, artifact_path="adapter")
    print(f"Adapter saved to: {adapter_path}")

    # ── Final output for orchestrator ────────────────────────
    print(f"wall_time: {wall_time:.1f}s")
    print(f"vram_peak_gb: {vram_peak_gb:.2f}")
    print(f"val_loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
