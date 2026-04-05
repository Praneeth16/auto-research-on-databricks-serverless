# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "transformers>=4.45.0",
#     "peft>=0.13.0",
#     "trl>=0.12.0",
#     "datasets>=3.0.0",
#     "accelerate>=0.34.0",
#     "bitsandbytes>=0.44.0",
#     "mlflow>=2.16.0",
# ]
# ///
"""
LoRA fine-tuning of Qwen 3.5 4B on financial text.

This file is the ONLY file modified by the auto-research agent.
Each experiment changes ONE parameter to isolate its effect on val_loss.

The agent can modify:
  - LoRA config (rank, alpha, dropout, target_modules)
  - Training hyperparams (lr, scheduler, warmup, batch_size, max_steps)
  - Data preprocessing (max_length, prompt template)
  - Optimization (gradient_checkpointing, bf16, optim)

The agent must NOT modify:
  - Base model loading (must remain Qwen 3.5 4B)
  - MLflow logging
  - The final val_loss print statement
  - Argument parsing
"""

import os
import sys
import time
import argparse

import torch
import mlflow
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


# ============================================================
# CONFIGURABLE PARAMETERS (auto-research agent modifies these)
# ============================================================

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = -1  # -1 means use num_train_epochs
NUM_TRAIN_EPOCHS = 1
OPTIM = "adamw_torch"

# Data preprocessing
MAX_LENGTH = 1024

# Optimization
USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = True
USE_4BIT = True  # QLoRA

# ============================================================
# BASE MODEL (do NOT modify)
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-3B"  # Qwen 3.5 4B when available, fallback to 2.5-3B


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./lora_output")
    parser.add_argument("--experiment-id", type=str, default="exp_0000")
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--results-file", type=str, default=None,
                        help="UC Volume path to write val_loss results for orchestrator pickup")
    return parser.parse_args()


def setup_quantization():
    """Configure 4-bit quantization for QLoRA."""
    if USE_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16" if USE_BF16 else "float16",
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model_and_tokenizer(quantization_config):
    """Load base model with quantization and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def create_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Configure SFTTrainer."""
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        optim=OPTIM,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",  # don't save checkpoints during auto-research
        max_length=MAX_LENGTH,
        report_to="mlflow",
        run_name=args.experiment_id,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if USE_GRADIENT_CHECKPOINTING else None,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    return trainer


def main():
    args = parse_args()

    print(f"Auto-research experiment: {args.experiment_id}")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Target modules: {LORA_TARGET_MODULES}")
    print(f"LR={LEARNING_RATE}, scheduler={LR_SCHEDULER_TYPE}, warmup={WARMUP_RATIO}")
    print(f"Batch size={PER_DEVICE_BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS}")
    print(f"Max length={MAX_LENGTH}, QLoRA={USE_4BIT}")
    print(f"Training budget: {args.max_seconds}s")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    dataset = load_from_disk(args.data_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    print(f"Train: {len(train_dataset)} examples, Val: {len(eval_dataset)} examples")

    # Load model
    print(f"\nLoading {BASE_MODEL}...")
    quant_config = setup_quantization()
    model, tokenizer = load_model_and_tokenizer(quant_config)

    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = setup_lora(model)

    # VRAM usage after model load
    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"VRAM after model load: {vram_gb:.2f} GB")

    # Train
    print("\nStarting training...")
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, args)

    # Add timeout callback
    from transformers import TrainerCallback

    class WallClockTimeout(TrainerCallback):
        def __init__(self, max_seconds):
            self.max_seconds = max_seconds
            self.start_time = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self.start_time and (time.time() - self.start_time) > self.max_seconds:
                control.should_training_stop = True

    trainer.add_callback(WallClockTimeout(args.max_seconds))

    train_result = trainer.train()

    # Evaluate
    print("\nRunning evaluation...")
    eval_result = trainer.evaluate()
    val_loss = eval_result["eval_loss"]

    # VRAM peak
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

    # Save LoRA adapter
    adapter_path = os.path.join(args.output_dir, args.experiment_id)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # Print metrics
    print(f"\ntrain_loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"train_steps: {train_result.metrics.get('train_steps', 'N/A')}")

    # THIS LINE MUST NOT BE MODIFIED - the orchestrator parses it
    print(f"val_loss: {val_loss}")

    # Write results to a file for orchestrator pickup.
    # SparkPythonTask stdout is not reliably accessible, so we write to
    # a known UC Volume path that the orchestrator reads back.
    if args.results_file:
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        results_content = (
            f"val_loss: {val_loss}\n"
            f"train_loss: {train_result.metrics.get('train_loss', 'N/A')}\n"
            f"peak_vram_gb: {peak_vram:.2f}\n"
            f"train_steps: {train_result.metrics.get('train_steps', 'N/A')}\n"
        )
        with open(args.results_file, "w") as f:
            f.write(results_content)
        print(f"Results written to {args.results_file}")


if __name__ == "__main__":
    main()
