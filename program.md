# Auto-Research Program: LoRA Fine-Tuning for Financial Text

## Objective

Minimize `val_loss` on a held-out financial text validation set by iteratively improving LoRA fine-tuning configuration. Each experiment is a single training run of Qwen 3.5 4B on a blended financial corpus (SEC 10-K filings, earnings call transcripts, financial news). You have 5 minutes of wall-clock time per run on a single A10G GPU (24 GB VRAM).

## Your task

You are an autonomous research agent. Before each experiment, you read `results.tsv` to see what has been tried. You then edit `train.py` to make exactly one change, run the experiment, and observe the resulting `val_loss`. The system handles logging and reverting failed experiments. Your job is to decide what to try next.

## What you can modify

You edit ONLY `train.py`. Within that file, you may change:

### LoRA configuration
- `r` (rank): integer from 4 to 128. Higher rank = more parameters = more expressive but slower and more VRAM.
- `lora_alpha`: integer from 8 to 256. Typical ratio is alpha = 2x rank.
- `lora_dropout`: float from 0.0 to 0.3.
- `target_modules`: any subset of `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. More modules = more parameters tuned.

### Training hyperparameters
- `learning_rate`: float from 1e-6 to 1e-3. This is the single most impactful hyperparameter.
- `lr_scheduler_type`: one of `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"constant"`, `"constant_with_warmup"`.
- `warmup_ratio`: float from 0.0 to 0.2.
- `weight_decay`: float from 0.0 to 0.1.
- `optim`: one of `"adamw_torch"`, `"adamw_8bit"`, `"paged_adamw_8bit"`, `"adafactor"`.

### Batch and sequence configuration
- `per_device_train_batch_size`: integer, must fit in 24 GB VRAM alongside the model (~16 GB for Qwen 3.5 4B with LoRA).
- `gradient_accumulation_steps`: integer, effective batch size = per_device_batch_size x gradient_accumulation_steps.
- `max_steps`: integer, constrained by 5-minute wall clock. Start conservative, increase if runs finish early.
- `max_length` (or `max_seq_length`): integer from 256 to 2048. Financial text is often long-form; longer sequences capture more context but use more VRAM.

### Optimization flags
- `gradient_checkpointing`: boolean. Saves VRAM at the cost of ~20% slower training. Enable if hitting OOM.
- `bf16` / `fp16`: use `bf16=True` on A10G (it supports bf16 natively). Only switch to fp16 if you have a specific reason.

### Data preprocessing
- Prompt template format: how the instruction/input/output fields are assembled into the training string. Financial text may benefit from structured templates that preserve document metadata.

## What you CANNOT modify

These are fixed. Do not change them under any circumstances:
- **Base model**: must remain `Qwen/Qwen2.5-3B` (the Qwen 3.5 4B model)
- **Dataset**: loaded from a fixed path. Do not change the data loading logic or the data files.
- **Evaluation**: val_loss is computed on the held-out split. Do not change how val_loss is calculated.
- **MLflow logging**: must remain enabled via `report_to="mlflow"`.
- **Final output**: the last line of stdout must print `val_loss: {value}`. Do not alter this print statement.

## Strategy

Follow these principles strictly:

### 1. One change per experiment
Make exactly one modification per run. If you change learning rate AND LoRA rank simultaneously, you cannot attribute the result to either change. Scientific method applies.

### 2. Prioritize high-impact hyperparameters first
Run experiments in roughly this order:
1. **Learning rate** (try 2e-4, 5e-4, 1e-4, 5e-5 — this usually dominates)
2. **LoRA rank** (try 8, 16, 32, 64 — diminishing returns above 32 for 4B models)
3. **Target modules** (start with `["q_proj", "v_proj"]`, then try adding `k_proj`, `o_proj`, then the MLP modules)
4. **Sequence length** (try 512, 1024, 2048 — financial filings are verbose, longer may help)
5. **Scheduler and warmup** (cosine with 5-10% warmup is a strong default)
6. **Batch size and gradient accumulation** (tune effective batch size: 8, 16, 32)
7. **Optimizer** (try `paged_adamw_8bit` to save VRAM, or `adafactor` for a different optimization trajectory)

### 3. Interpret results relative to baselines
- If val_loss improves: keep the change and build on it.
- If val_loss regresses: revert and try a different direction.
- If val_loss is unchanged (within 0.5% of previous best): the change had no effect. Revert and try something with more leverage.

### 4. Handle failures gracefully
- **OOM error**: reduce `per_device_train_batch_size` by half, or enable `gradient_checkpointing=True`, or reduce `max_length`.
- **Training instability (loss spikes or NaN)**: reduce `learning_rate` by 5x or add `warmup_ratio=0.1`.
- **Run exceeds 5 minutes**: reduce `max_steps` or reduce `max_length` to increase throughput.

### 5. Domain-specific considerations
- Financial text contains specialized vocabulary (EBITDA, diluted EPS, Form 10-K). The model benefits from seeing full contexts rather than aggressively truncated sequences.
- Earnings call transcripts have a Q&A structure. A prompt template that preserves the question/answer boundary may help.
- SEC filings use formal, repetitive language. The model may learn these patterns quickly, meaning fewer steps at a higher learning rate could suffice.

## Results format

After each run, the system appends a row to `results.tsv` with these columns:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 timestamp of run completion |
| `experiment_id` | Unique run identifier |
| `description` | Your one-line description of what changed |
| `val_loss` | The scalar validation loss (lower is better) |
| `kept` | `true` if val_loss improved over previous best, `false` if reverted |

Read this file before each experiment. Use it to understand the trajectory and avoid repeating failed experiments.

## Example decision process

1. Read `results.tsv`. The best val_loss so far is 2.31 from experiment 5 (lr=2e-4, rank=16, target_modules=["q_proj", "v_proj"]).
2. Hypothesis: increasing LoRA rank from 16 to 32 adds capacity that could reduce val_loss further.
3. Edit `train.py`: change `r=16` to `r=32`, and `lora_alpha=32` to `lora_alpha=64` (maintaining 2x ratio).
4. Run the experiment.
5. Result: val_loss=2.25. Improvement. Keep the change.
6. Next experiment: try adding `k_proj` and `o_proj` to target_modules.
