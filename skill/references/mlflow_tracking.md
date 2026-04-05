# MLflow Tracking for Auto-Research Training

## Integration with SFTTrainer

The simplest integration path is `report_to="mlflow"` in `SFTConfig`. This automatically
logs training metrics, hyperparameters, and model artifacts to the active MLflow experiment.

```python
import mlflow
from trl import SFTConfig

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{username}/autoresearch/{experiment_id}")

training_config = SFTConfig(
    output_dir="./output",
    report_to="mlflow",
    run_name=f"autoresearch-{experiment_id}-run-{run_number}",
    # ... other training args
)
```

## What Gets Logged Automatically

When `report_to="mlflow"` is set, `SFTTrainer` logs:

| Metric             | Logged at           | Source          |
|-------------------|---------------------|-----------------|
| `train/loss`      | Every `logging_steps`| Trainer         |
| `eval/loss`       | Every `eval_steps`   | Trainer         |
| `train/learning_rate` | Every `logging_steps`| Scheduler   |
| `train/epoch`     | Every `logging_steps`| Trainer         |
| `train/global_step`| Every `logging_steps`| Trainer        |

## Custom Metrics to Log

The auto-research agent also logs these metrics manually:

### VRAM Peak Usage

```python
import torch

def log_vram_peak():
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        mlflow.log_metric("vram_peak_gb", round(peak_gb, 2))
        torch.cuda.reset_peak_memory_stats()
```

### Throughput (tokens/sec)

```python
import time

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

total_tokens = len(train_dataset) * training_config.max_length * training_config.max_steps
tokens_per_sec = total_tokens / elapsed
mlflow.log_metric("tokens_per_sec", round(tokens_per_sec, 1))
mlflow.log_metric("wall_time_seconds", round(elapsed, 1))
```

### Final Validation Loss

```python
eval_result = trainer.evaluate()
val_loss = eval_result["eval_loss"]
mlflow.log_metric("val_loss", val_loss)

# Print for orchestrator parsing
print(f"val_loss: {val_loss:.6f}")
```

## Logging Artifacts

### Training Script Snapshot

Capture the exact script that produced this run:

```python
mlflow.log_artifact("train_sft_lora.py", artifact_path="scripts")
```

### LoRA Adapter Weights

```python
adapter_path = "/Volumes/catalog/schema/autoresearch/adapters/{run_id}"
trainer.save_model(adapter_path)
mlflow.log_artifacts(adapter_path, artifact_path="adapter")
```

### Training Configuration

```python
import json

config = {
    "base_model": base_model,
    "lora_rank": lora_rank,
    "learning_rate": learning_rate,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "use_qlora": use_qlora,
    "dataset_path": dataset_path,
}
with open("/tmp/training_config.json", "w") as f:
    json.dump(config, f, indent=2)
mlflow.log_artifact("/tmp/training_config.json", artifact_path="config")
```

## Logging Tags

Tags provide searchable metadata for filtering experiments in the MLflow UI.

```python
mlflow.set_tags({
    "experiment_description": "Testing rank=32 with cosine scheduler on filtered dataset",
    "keep_or_revert": "pending",      # updated by orchestrator after evaluation
    "base_model": "Qwen/Qwen2.5-3B",
    "dataset_hash": dataset_hash,     # SHA256 of the training data
    "training_method": "qlora_sft",
    "auto_research_iteration": str(iteration_number),
})
```

After the orchestrator evaluates the run:

```python
mlflow.set_tag("keep_or_revert", "keep")  # or "revert"
mlflow.set_tag("decision_reason", "val_loss improved from 1.82 to 1.74")
```

## Experiment Naming Convention

```
/Users/{username}/autoresearch/{experiment_id}
```

Where:
- `username` is the Databricks workspace user (e.g., `praneeth@databricks.com`)
- `experiment_id` is a slug like `qwen3.5-4b-code-review` describing the research goal

Each run within the experiment is named:
```
autoresearch-{experiment_id}-run-{N}
```

## Querying Past Runs

The agent uses the MLflow API to analyze previous experiments and decide next steps:

```python
import mlflow

experiment = mlflow.get_experiment_by_name(f"/Users/{username}/autoresearch/{experiment_id}")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_loss ASC"],
    max_results=10,
)

best_run = runs.iloc[0]
best_val_loss = best_run["metrics.val_loss"]
best_params = {
    "lora_rank": best_run["params.lora_rank"],
    "learning_rate": best_run["params.learning_rate"],
}
```

## Environment Variables

Set these before training to ensure MLflow connects to the right place:

```bash
export MLFLOW_TRACKING_URI="databricks"
export MLFLOW_EXPERIMENT_NAME="/Users/{username}/autoresearch/{experiment_id}"
export HF_MLFLOW_LOG_ARTIFACTS=1
```

On Databricks clusters, `MLFLOW_TRACKING_URI="databricks"` is set by default.
`HF_MLFLOW_LOG_ARTIFACTS=1` tells the HuggingFace MLflow integration to log
model artifacts (not just metrics).
