# SFT with LoRA Training Methods

## Overview

The auto-research agent uses Supervised Fine-Tuning (SFT) with Low-Rank Adaptation (LoRA)
to adapt base language models. The stack is `trl.SFTTrainer` + `peft.LoraConfig` + optional
4-bit quantization via `bitsandbytes` (QLoRA).

## LoRA Configuration

LoRA injects low-rank decomposition matrices into attention layers, training only a small
fraction of total parameters while keeping the base model frozen.

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more VRAM
    lora_alpha=32,                 # scaling factor, typically 2x rank
    target_modules=[               # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,             # regularization
    bias="none",                   # don't train bias terms
    task_type=TaskType.CAUSAL_LM,
)
```

### Parameter Guidelines

| Parameter        | Auto-Research Default | Range        | Effect                            |
|-----------------|----------------------|--------------|-----------------------------------|
| `r` (rank)      | 16                   | 4-64         | Higher = more capacity, slower    |
| `lora_alpha`    | 32                   | 8-128        | Scaling; typically `2 * r`        |
| `lora_dropout`  | 0.05                 | 0.0-0.1      | Regularization for small datasets |
| `target_modules`| All linear layers    | See below    | More modules = more trainable params |

### Target Modules by Architecture

For Qwen models, target all linear projection layers:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

For Llama models, same set applies. For models with different naming, inspect:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)
print([name for name, _ in model.named_modules() if "proj" in name or "dense" in name])
```

## SFTTrainer Configuration

Use `trl.SFTConfig` (not the deprecated `TrainingArguments`) to configure the trainer.

```python
from trl import SFTConfig, SFTTrainer

training_config = SFTConfig(
    output_dir="./output",
    max_steps=200,                     # fixed step count for auto-research
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,     # effective batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    bf16=True,                         # A10G supports BF16
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    max_length=1024,                   # NOTE: max_length, NOT max_seq_length
    gradient_checkpointing=True,       # saves ~30% VRAM
    report_to="mlflow",
    run_name="autoresearch-experiment",
    dataloader_num_workers=2,
    optim="paged_adamw_8bit",          # memory-efficient optimizer
)
```

**Important**: `SFTConfig` uses `max_length` (not `max_seq_length`). Using the wrong
parameter name silently defaults to 1024 and your sequences may be truncated unexpectedly.

### Assembling the Trainer

```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B",
    trust_remote_code=True,
)

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()
```

## QLoRA (4-bit Quantization)

QLoRA loads the base model in 4-bit precision, reducing VRAM by ~70% while training
LoRA adapters in FP16/BF16. Essential for 7B+ models on A10G.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # normalized float 4-bit
    bnb_4bit_compute_dtype="bfloat16",   # compute in BF16 for speed
    bnb_4bit_use_double_quant=True,      # nested quantization saves ~0.4 GB
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    quantization_config=bnb_config,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)
```

### When to Use QLoRA vs LoRA

| Scenario                        | Use       | Why                                |
|--------------------------------|-----------|-------------------------------------|
| Model <= 3B on A10G            | LoRA      | Fits in 24 GB without quantization |
| Model 4-7B on A10G             | QLoRA     | Needs 4-bit to fit                 |
| Model 7B+ on A10G              | QLoRA     | Mandatory                          |
| Maximizing training quality    | LoRA      | No quantization noise              |
| Maximizing speed on small model| LoRA      | No quantization overhead           |

## Dataset Format

SFTTrainer expects one of these formats:

### Conversational (preferred for chat models)
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Text column
```json
{"text": "### Instruction\n...\n### Response\n..."}
```

The auto-research agent should produce datasets in conversational format since Qwen
models are chat-tuned. Use `dataset_inspector.py` to validate before training.

## How the Auto-Research Agent Parameterizes Training

The agent modifies `train_sft_lora.py` by injecting a config dict at the top of the script:

```python
CONFIG = {
    "base_model": "Qwen/Qwen2.5-3B",
    "dataset_path": "/Volumes/catalog/schema/autoresearch/data/experiment_42.jsonl",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "max_steps": 200,
    "batch_size": 4,
    "max_length": 1024,
    "use_qlora": True,
    "gradient_checkpointing": True,
}
```

The orchestrator then submits this modified script as a Databricks job.
