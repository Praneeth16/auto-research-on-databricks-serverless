---
name: databricks-lora-trainer
description: >
  Use this skill when the auto-research agent needs to submit and manage
  LoRA fine-tuning jobs on Databricks serverless GPUs. Covers dataset
  validation, hardware selection, SFT training with trl/peft, MLflow
  experiment tracking, and LoRA adapter persistence to Unity Catalog Volumes.
version: 0.1.0
tags: [fine-tuning, lora, databricks, mlflow, auto-research]
---

# databricks-lora-trainer

Submit and manage LoRA fine-tuning jobs on Databricks serverless GPUs as part of the auto-research loop.

## Key Directives

1. **Use `databricks-sdk`** to submit training jobs programmatically. Never rely on manual cluster creation.
2. **Log all metrics to MLflow**: `val_loss`, `train_loss`, VRAM usage, throughput (tokens/sec).
3. **Persist LoRA adapters** to UC Volumes at `/Volumes/catalog/schema/autoresearch/adapters/{run_id}/`.
4. **Validate datasets before GPU burn** using `dataset_inspector.py`. A failed validation must block training submission.
5. **Select GPU node type** based on model parameter count. See the hardware selection table below.
6. **Set a 5-minute timeout** for auto-research experiments. If the job exceeds this, it is killed and logged as a timeout.
7. **Print `val_loss: {value}`** as the last stdout line so the orchestrator can parse the scalar result.

## Quick Start

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, PythonWheelTask, NotebookTask
import json

w = WorkspaceClient()

# Submit a LoRA fine-tuning run
run = w.jobs.submit(
    run_name="autoresearch-lora-qwen3.5-4b",
    tasks=[
        SubmitTask(
            task_key="train",
            new_cluster={
                "spark_version": "15.4.x-gpu-ml-scala2.12",
                "node_type_id": "g5.xlarge",
                "num_workers": 0,
                "spark_conf": {"spark.master": "local[*]"},
            },
            spark_python_task={
                "python_file": "/Volumes/catalog/schema/autoresearch/scripts/train_sft_lora.py",
                "parameters": json.dumps({
                    "base_model": "Qwen/Qwen2.5-3B",
                    "dataset_path": "/Volumes/catalog/schema/autoresearch/data/train.jsonl",
                    "lora_rank": 16,
                    "learning_rate": 2e-4,
                    "max_steps": 200,
                }),
            },
            timeout_seconds=300,
        )
    ],
).result()
```

## Hardware Selection Table

| Model Size | Node Type      | GPU              | VRAM   | DBU/hr | Notes                              |
|-----------|----------------|------------------|--------|--------|------------------------------------|
| 1-3B      | `g5.xlarge`    | 1x A10G          | 24 GB  | ~4.0   | Comfortable with QLoRA             |
| 3-7B      | `g5.xlarge`    | 1x A10G          | 24 GB  | ~4.0   | Needs QLoRA (4-bit) for 7B        |
| 7-13B     | `g5.2xlarge`   | 1x A10G          | 24 GB  | ~8.0   | QLoRA required, more CPU/RAM       |
| 13B+      | `p4d.24xlarge` | 8x A100 (40 GB)  | 320 GB | ~65.0  | Overkill for auto-research scope   |

**Memory formula**: LoRA fine-tuning VRAM ~= `(params_in_billions) x 4 GB`.
For Qwen 3.5 4B with LoRA: ~16 GB VRAM, fits on A10G (24 GB) with headroom.

## MLflow Integration

Every training run automatically logs to MLflow via `report_to="mlflow"` in `SFTConfig`.

- **Experiment name**: `autoresearch/{experiment_id}`
- **Logged metrics**: `train_loss`, `val_loss`, `learning_rate`, `epoch`, `vram_peak_gb`, `tokens_per_sec`
- **Logged artifacts**: `train.py` snapshot, LoRA adapter directory, training config JSON
- **Logged tags**: `experiment_description`, `keep_or_revert`, `base_model`, `dataset_hash`

## Cost Estimation

Use `scripts/estimate_cost.py` to predict costs before submitting. Typical auto-research run:

- **Qwen 3.5 4B, 1K examples, 200 steps on g5.xlarge**: ~0.33 DBU (~$0.23 at $0.70/DBU)
- **Budget guard**: The orchestrator should reject runs estimated above 2.0 DBU (~$1.40)

## Reference Documents

- `references/hardware_guide.md` — GPU node specs and selection criteria
- `references/training_methods.md` — SFT + LoRA configuration with trl/peft
- `references/mlflow_tracking.md` — MLflow experiment tracking setup
- `references/troubleshooting.md` — Common failure modes and fixes

## Scripts

- `scripts/train_sft_lora.py` — Base training template (the agent modifies this per experiment)
- `scripts/dataset_inspector.py` — Pre-flight dataset validation
- `scripts/estimate_cost.py` — DBU cost estimator
