# Databricks GPU Node Types for LoRA Fine-Tuning

## Available Node Types

### g5.xlarge — Default for Auto-Research

| Spec         | Value            |
|-------------|------------------|
| GPU          | 1x NVIDIA A10G   |
| VRAM         | 24 GB GDDR6      |
| vCPUs        | 4                |
| RAM          | 16 GB            |
| DBU/hr       | ~4.0             |
| Storage      | 250 GB NVMe SSD  |
| Best for     | 1-7B models with QLoRA, all auto-research experiments |

The A10G supports BF16, FP16, TF32, and INT8. It has 31.2 TFLOPS FP32 and
125 TFLOPS TF32, making it efficient for mixed-precision LoRA training.

### g5.2xlarge — When You Need More CPU/RAM

| Spec         | Value            |
|-------------|------------------|
| GPU          | 1x NVIDIA A10G   |
| VRAM         | 24 GB GDDR6      |
| vCPUs        | 8                |
| RAM          | 32 GB            |
| DBU/hr       | ~8.0             |
| Storage      | 450 GB NVMe SSD  |
| Best for     | 7-13B models with QLoRA, large datasets needing CPU preprocessing |

Same GPU as g5.xlarge but double the CPU and system RAM. Only use when
dataset preprocessing is the bottleneck or the model has large embedding
tables that spill to CPU.

### p3.2xlarge — Legacy Fallback

| Spec         | Value            |
|-------------|------------------|
| GPU          | 1x NVIDIA V100   |
| VRAM         | 16 GB HBM2       |
| vCPUs        | 8                |
| RAM          | 61 GB            |
| DBU/hr       | ~5.5             |
| Storage      | EBS only         |
| Best for     | Fallback when g5 is unavailable |

The V100 lacks BF16 support. Use FP16 only. At 16 GB VRAM, only 1-3B
models fit comfortably with QLoRA. More expensive per VRAM-GB than A10G.
Avoid unless g5 instances are unavailable in your region.

### p4d.24xlarge — Multi-GPU (Out of Scope)

| Spec         | Value             |
|-------------|-------------------|
| GPU          | 8x NVIDIA A100    |
| VRAM         | 8x 40 GB (320 GB) |
| vCPUs        | 96                |
| RAM          | 1152 GB           |
| DBU/hr       | ~65.0             |
| Best for     | Full fine-tuning of 13B+ models |

Massively overkill for auto-research LoRA experiments. Each run would
cost ~$7.50/hr. Only relevant for final production training after
auto-research has converged on the best hyperparameters.

## VRAM Estimation Formula

```
LoRA fine-tuning VRAM (GB) ≈ (model_params_in_billions) × 4

With QLoRA (4-bit quantization):
QLoRA VRAM (GB) ≈ (model_params_in_billions) × 1.2 + 2 (overhead)
```

### Examples

| Model               | Params | LoRA VRAM | QLoRA VRAM | Fits A10G (24GB)? |
|---------------------|--------|-----------|------------|--------------------|
| Qwen2.5-1.5B       | 1.5B   | ~6 GB     | ~3.8 GB    | Yes (LoRA or QLoRA)|
| Qwen2.5-3B         | 3B     | ~12 GB    | ~5.6 GB    | Yes (LoRA or QLoRA)|
| Qwen3.5-4B         | 4B     | ~16 GB    | ~6.8 GB    | Yes (LoRA or QLoRA)|
| Qwen2.5-7B         | 7B     | ~28 GB    | ~10.4 GB   | QLoRA only         |
| Llama-3.1-8B       | 8B     | ~32 GB    | ~11.6 GB   | QLoRA only         |
| Qwen2.5-14B        | 14B    | ~56 GB    | ~18.8 GB   | QLoRA only (tight) |

## Selection Logic for Auto-Research Agent

```python
def select_node_type(model_params_billions: float, use_qlora: bool = True) -> str:
    if use_qlora:
        vram_needed = model_params_billions * 1.2 + 2
    else:
        vram_needed = model_params_billions * 4

    if vram_needed <= 22:  # 24 GB with 2 GB headroom
        return "g5.xlarge"
    elif vram_needed <= 22 and model_params_billions > 7:
        return "g5.2xlarge"  # more CPU for larger models
    else:
        raise ValueError(
            f"Model requires ~{vram_needed:.0f} GB VRAM. "
            "Exceeds single-GPU capacity. Use p4d.24xlarge or reduce model size."
        )
```

## Cluster Configuration

When submitting via `databricks-sdk`, use this cluster spec:

```python
new_cluster = {
    "spark_version": "15.4.x-gpu-ml-scala2.12",
    "node_type_id": "g5.xlarge",
    "num_workers": 0,  # single-node for LoRA
    "spark_conf": {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
    },
    "custom_tags": {
        "ResourceClass": "SingleNode",
        "project": "autoresearch",
    },
}
```

Key points:
- `num_workers: 0` for single-node training (LoRA does not need distributed compute)
- Use `15.4.x-gpu-ml-scala2.12` runtime which bundles PyTorch, CUDA, and ML libs
- The `-gpu-ml-` variant includes pre-installed `transformers`, `peft`, `trl`, `bitsandbytes`
