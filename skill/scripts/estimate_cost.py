# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Estimate Databricks training cost for a LoRA fine-tuning run.

Provides DBU consumption, wall time, and dollar cost estimates based on
model size, dataset size, hardware type, and training configuration.

Usage:
    python estimate_cost.py --model Qwen/Qwen2.5-3B --dataset-size 1000 \
        --max-steps 200 --hardware g5.xlarge
    python estimate_cost.py --model-params 4.0 --dataset-size 5000 \
        --num-epochs 3 --hardware g5.xlarge
"""

import argparse
import json
import math

# Databricks node specs: (DBU/hr, GPU count, VRAM per GPU in GB)
NODE_SPECS = {
    "g5.xlarge":     {"dbu_per_hr": 4.0,  "gpus": 1, "vram_gb": 24, "gpu_name": "A10G"},
    "g5.2xlarge":    {"dbu_per_hr": 8.0,  "gpus": 1, "vram_gb": 24, "gpu_name": "A10G"},
    "p3.2xlarge":    {"dbu_per_hr": 5.5,  "gpus": 1, "vram_gb": 16, "gpu_name": "V100"},
    "p4d.24xlarge":  {"dbu_per_hr": 65.0, "gpus": 8, "vram_gb": 40, "gpu_name": "A100"},
}

# Approximate model parameter counts for common models
MODEL_PARAMS = {
    "Qwen/Qwen2.5-1.5B": 1.5,
    "Qwen/Qwen2.5-3B": 3.0,
    "Qwen/Qwen3.5-4B": 4.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "Qwen/Qwen2.5-14B": 14.0,
    "meta-llama/Llama-3.1-8B": 8.0,
    "meta-llama/Llama-3.2-3B": 3.0,
    "mistralai/Mistral-7B-v0.3": 7.0,
}

DEFAULT_DBU_PRICE = 0.70  # $/DBU for jobs compute (varies by plan/region)

# Empirical throughput: tokens/sec per GPU for QLoRA SFT with batch_size=4
# Measured on A10G with gradient checkpointing enabled
THROUGHPUT_TOKENS_PER_SEC = {
    "A10G": {1.5: 3200, 3.0: 1800, 4.0: 1400, 7.0: 800, 14.0: 350},
    "V100": {1.5: 2400, 3.0: 1200, 4.0: 900, 7.0: 500, 14.0: 0},
    "A100": {1.5: 6000, 3.0: 4000, 4.0: 3200, 7.0: 2000, 14.0: 1200},
}


def get_model_params(model_name: str | None, model_params: float | None) -> float:
    if model_params is not None:
        return model_params
    if model_name and model_name in MODEL_PARAMS:
        return MODEL_PARAMS[model_name]
    raise ValueError(
        f"Unknown model '{model_name}'. Provide --model-params explicitly."
    )


def estimate_throughput(gpu_name: str, params_b: float) -> float:
    thresholds = THROUGHPUT_TOKENS_PER_SEC.get(gpu_name, {})
    if not thresholds:
        return 1000.0  # conservative default

    closest = min(thresholds.keys(), key=lambda p: abs(p - params_b))
    base = thresholds[closest]
    if base == 0:
        return 0.0

    # Linear interpolation adjustment
    ratio = closest / params_b if params_b > 0 else 1.0
    return max(base * math.sqrt(ratio), 100.0)


def estimate(
    model_name: str | None = None,
    model_params: float | None = None,
    dataset_size: int = 1000,
    max_steps: int | None = None,
    num_epochs: int | None = None,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    max_length: int = 1024,
    hardware: str = "g5.xlarge",
    dbu_price: float = DEFAULT_DBU_PRICE,
    cluster_startup_minutes: float = 3.0,
) -> dict:
    params_b = get_model_params(model_name, model_params)
    node = NODE_SPECS.get(hardware)
    if node is None:
        raise ValueError(f"Unknown hardware type '{hardware}'. Options: {list(NODE_SPECS)}")

    # Check VRAM fit
    qlora_vram = params_b * 1.2 + 2
    fits_qlora = qlora_vram <= node["vram_gb"]
    lora_vram = params_b * 4
    fits_lora = lora_vram <= node["vram_gb"]

    if not fits_qlora:
        return {
            "error": (
                f"Model ({params_b}B) requires ~{qlora_vram:.0f} GB VRAM with QLoRA, "
                f"but {hardware} has {node['vram_gb']} GB. Choose a larger node."
            )
        }

    effective_batch = batch_size * gradient_accumulation_steps
    tokens_per_step = effective_batch * max_length

    if max_steps:
        total_steps = max_steps
    elif num_epochs:
        steps_per_epoch = math.ceil(dataset_size / effective_batch)
        total_steps = steps_per_epoch * num_epochs
    else:
        total_steps = 200  # auto-research default

    total_tokens = total_steps * tokens_per_step
    throughput = estimate_throughput(node["gpu_name"], params_b)

    if throughput <= 0:
        return {"error": f"Model too large for {node['gpu_name']}"}

    training_seconds = total_tokens / throughput
    training_minutes = training_seconds / 60
    total_minutes = training_minutes + cluster_startup_minutes
    total_hours = total_minutes / 60

    dbus = total_hours * node["dbu_per_hr"]
    cost = dbus * dbu_price

    return {
        "model_params_b": params_b,
        "hardware": hardware,
        "gpu": f"{node['gpus']}x {node['gpu_name']} ({node['vram_gb']} GB)",
        "fits_lora": fits_lora,
        "fits_qlora": fits_qlora,
        "recommended_method": "lora" if fits_lora else "qlora",
        "estimated_vram_gb": round(lora_vram if fits_lora else qlora_vram, 1),
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "throughput_tokens_per_sec": round(throughput, 0),
        "training_minutes": round(training_minutes, 1),
        "cluster_startup_minutes": cluster_startup_minutes,
        "total_wall_minutes": round(total_minutes, 1),
        "dbu_per_hr": node["dbu_per_hr"],
        "estimated_dbus": round(dbus, 2),
        "dbu_price": dbu_price,
        "estimated_cost_usd": round(cost, 2),
        "within_5min_budget": total_minutes <= 5.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate Databricks LoRA training cost")
    parser.add_argument("--model", default=None, help="HuggingFace model name")
    parser.add_argument("--model-params", type=float, default=None, help="Model size in billions")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Number of training examples")
    parser.add_argument("--max-steps", type=int, default=None, help="Training steps (overrides epochs)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--hardware", default="g5.xlarge", help="Databricks node type")
    parser.add_argument("--dbu-price", type=float, default=DEFAULT_DBU_PRICE, help="$/DBU")
    parser.add_argument("--startup-minutes", type=float, default=3.0, help="Cluster startup time")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = estimate(
        model_name=args.model,
        model_params=args.model_params,
        dataset_size=args.dataset_size,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        hardware=args.hardware,
        dbu_price=args.dbu_price,
        cluster_startup_minutes=args.startup_minutes,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\n{'='*50}")
    print(f"Cost Estimate: LoRA Fine-Tuning")
    print(f"{'='*50}")
    print(f"Model:            {args.model or f'{args.model_params}B params'}")
    print(f"Hardware:         {result['hardware']} ({result['gpu']})")
    print(f"Method:           {result['recommended_method'].upper()}")
    print(f"Est. VRAM:        {result['estimated_vram_gb']} GB")
    print(f"")
    print(f"Training steps:   {result['total_steps']}")
    print(f"Total tokens:     {result['total_tokens']:,}")
    print(f"Throughput:       {result['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"")
    print(f"Training time:    {result['training_minutes']:.1f} min")
    print(f"Startup time:     {result['cluster_startup_minutes']:.1f} min")
    print(f"Total wall time:  {result['total_wall_minutes']:.1f} min")
    print(f"")
    print(f"DBU rate:         {result['dbu_per_hr']} DBU/hr")
    print(f"Est. DBUs:        {result['estimated_dbus']:.2f}")
    print(f"Est. cost:        ${result['estimated_cost_usd']:.2f} (at ${args.dbu_price}/DBU)")
    print(f"")
    budget_status = "YES" if result["within_5min_budget"] else "NO"
    print(f"Within 5-min budget: {budget_status}")


if __name__ == "__main__":
    main()
