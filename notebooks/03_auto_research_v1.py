# Databricks notebook source
# MAGIC %md
# MAGIC # Auto-Research Loop
# MAGIC Autonomous LoRA fine-tuning optimization on Databricks.
# MAGIC
# MAGIC The agent (Llama 3.3 70B via FM API) proposes changes to training config,
# MAGIC we run each experiment for 5 min, keep improvements, revert failures.

# COMMAND ----------

import os, sys, time, json, copy, csv, difflib
import torch
import numpy as np
from datetime import datetime, timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

MAX_EXPERIMENTS = 20
TRAINING_BUDGET_SECONDS = 300  # 5 min per experiment
AGENT_MODEL = "databricks-meta-llama-3-3-70b-instruct"
DATA_PATH = "/Volumes/main/auto_research/autoresearch/data/financial_instruct"
ADAPTER_BASE = "/Volumes/main/auto_research/autoresearch/adapters"

# FM API auth - set early so call_agent can use them
import requests as _requests_lib
DB_HOST = spark.conf.get("spark.databricks.workspaceUrl", "")
if not DB_HOST.startswith("http"): DB_HOST = f"https://{DB_HOST}"
DB_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
print(f"FM API: host={DB_HOST}, token={'present' if DB_TOKEN else 'MISSING'}")
RESULTS_FILE = "/tmp/auto_research_results.tsv"  # local tmp, copy to UC Volumes at end
UC_RESULTS_FILE = "/Volumes/main/auto_research/autoresearch/auto_research_results.tsv"
BASE_MODEL = "Qwen/Qwen2.5-3B"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Configuration (what the agent modifies)

# COMMAND ----------

# Default config - the agent will propose modifications to this
DEFAULT_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 1024,
    "gradient_checkpointing": True,
    "bf16": True,
    "optim": "adamw_torch",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Function

# COMMAND ----------

def run_training(config, experiment_id):
    """Run a single LoRA training experiment and return val_loss."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_from_disk

    print(f"\n--- Training {experiment_id} ---")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    ds = load_from_disk(DATA_PATH)

    # Load model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config,
        device_map="auto", trust_remote_code=True,
    )

    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Timeout callback
    class WallClockTimeout(TrainerCallback):
        def __init__(self, max_seconds):
            self.max_seconds = max_seconds
            self.start_time = None
        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
        def on_step_end(self, args, state, control, **kwargs):
            if self.start_time and (time.time() - self.start_time) > self.max_seconds:
                control.should_training_stop = True

    # Training args
    training_args = SFTConfig(
        output_dir=f"/tmp/autoresearch/{experiment_id}",
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        num_train_epochs=1,
        optim=config["optim"],
        bf16=config["bf16"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        max_length=config["max_length"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config["gradient_checkpointing"] else None,
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=training_args,
    )
    trainer.add_callback(WallClockTimeout(TRAINING_BUDGET_SECONDS))

    # Train
    start = time.time()
    train_result = trainer.train()
    train_time = time.time() - start

    # Eval
    eval_result = trainer.evaluate()
    val_loss = eval_result["eval_loss"]
    train_loss = train_result.metrics.get("train_loss", None)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Save adapter
    adapter_path = f"{ADAPTER_BASE}/{experiment_id}"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

    result = {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "train_time_seconds": round(train_time, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "adapter_path": adapter_path,
    }
    print(f"Result: val_loss={val_loss:.6f}, train_loss={train_loss}, vram={peak_vram:.1f}GB, time={train_time:.0f}s")
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent LLM

# COMMAND ----------

def call_agent(current_config, history):
    """Ask the agent to propose a config change using Databricks SDK."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    system = """You are an ML researcher optimizing LoRA fine-tuning. Propose ONE config change.
Reply with ONLY this JSON (no other text):
{"param": "parameter_name", "value": new_value, "description": "what you changed", "hypothesis": "expected effect"}

Parameters you can change:
learning_rate (float, 1e-6 to 1e-3), lora_rank (int, 4-128), lora_alpha (int, 8-256),
lora_dropout (float, 0-0.3), lr_scheduler_type (str), warmup_ratio (float, 0-0.2),
weight_decay (float, 0-0.1), per_device_batch_size (int, 1-8), gradient_accumulation_steps (int, 1-16),
max_length (int, 256-2048), optim (str: adamw_torch/adamw_8bit/paged_adamw_8bit/adafactor)"""

    user = f"Current: lr={current_config['learning_rate']}, rank={current_config['lora_rank']}, alpha={current_config['lora_alpha']}, dropout={current_config['lora_dropout']}, modules={current_config['target_modules']}, scheduler={current_config['lr_scheduler_type']}, batch={current_config['per_device_batch_size']}, grad_accum={current_config['gradient_accumulation_steps']}, max_len={current_config['max_length']}, optim={current_config['optim']}\n\nHistory:\n{history}\n\nPropose ONE change. Reply with ONLY JSON."

    try:
        import requests
        api_url = f"{DB_HOST}/serving-endpoints/{AGENT_MODEL}/invocations"
        r = requests.post(api_url,
            json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 256, "temperature": 0.7},
            headers={"Authorization": f"Bearer {DB_TOKEN}", "Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print(f"Agent response: {text[:200]}")

        # Parse simple JSON
        if "```" in text: text = text.split("```")[1].split("```")[0]
        if text.strip().startswith("json"): text = text.strip()[4:]
        start = text.find("{"); end = text.rfind("}") + 1
        parsed = json.loads(text[start:end])

        # Apply the single param change to current config
        param = parsed["param"]
        value = parsed["value"]
        new_config = copy.deepcopy(current_config)
        if param in new_config:
            new_config[param] = value
            # Keep alpha = 2x rank convention
            if param == "lora_rank":
                new_config["lora_alpha"] = value * 2
        return {"config": new_config, "description": parsed.get("description", f"Changed {param} to {value}"), "hypothesis": parsed.get("hypothesis", "")}
    except Exception as e:
        import traceback
        print(f"Agent error: {e}")
        traceback.print_exc()
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Loop

# COMMAND ----------

# Pre-flight: verify agent LLM works before burning GPU time
print("Testing agent LLM call...")
test_result = call_agent(DEFAULT_CONFIG, "No experiments yet.")
if test_result is None:
    # Try a raw API call to diagnose
    _test_url = f"{DB_HOST}/serving-endpoints/{AGENT_MODEL}/invocations"
    _test_r = _requests_lib.post(_test_url, json={"messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10},
                        headers={"Authorization": f"Bearer {DB_TOKEN}", "Content-Type": "application/json"}, timeout=30)
    raise RuntimeError(f"Agent call_agent returned None. Raw API status={_test_r.status_code}, body={_test_r.text[:300]}")
print(f"Agent test OK: {test_result.get('description', 'no description')}")
print(f"Agent test OK: {test_result.get('description', 'no description')}")
print(f"Proposed config change: {json.dumps({k:v for k,v in test_result.get('config',{}).items() if str(v) != str(DEFAULT_CONFIG.get(k))}, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Loop

# COMMAND ----------

best_val_loss = float("inf")
best_config = copy.deepcopy(DEFAULT_CONFIG)
current_config = copy.deepcopy(DEFAULT_CONFIG)
history_lines = ["experiment_id\tdescription\tval_loss\tkept"]

# Write header to results file
with open(RESULTS_FILE, "w") as f:
    f.write("timestamp\texperiment_id\tdescription\thypothesis\tval_loss\ttrain_loss\tpeak_vram_gb\tkept\n")

print(f"Starting auto-research loop: {MAX_EXPERIMENTS} experiments, {TRAINING_BUDGET_SECONDS}s each")
print(f"Agent: {AGENT_MODEL}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

for exp_num in range(MAX_EXPERIMENTS):
    experiment_id = f"autoresearch_{exp_num:03d}"
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_num}/{MAX_EXPERIMENTS}")
    print(f"Best val_loss so far: {best_val_loss}")
    print(f"{'='*60}")

    # Step 1: Get agent proposal
    history_str = "\n".join(history_lines[-15:])  # last 15 experiments
    if exp_num == 0:
        # First run: use defaults
        proposal_config = current_config
        description = "Baseline run with default config"
        hypothesis = "Establish baseline val_loss"
    else:
        print("Calling agent LLM...")
        proposal = call_agent(current_config, history_str)
        if proposal is None:
            print("Agent failed, using predefined schedule")
            # Predefined experiment schedule instead of random
            SCHEDULE = [
                {"param": "learning_rate", "value": 5e-4},
                {"param": "learning_rate", "value": 1e-4},
                {"param": "learning_rate", "value": 5e-5},
                {"param": "lora_rank", "value": 32},
                {"param": "lora_rank", "value": 64},
                {"param": "max_length", "value": 512},
                {"param": "max_length", "value": 2048},
                {"param": "lora_dropout", "value": 0.1},
                {"param": "per_device_batch_size", "value": 2},
                {"param": "gradient_accumulation_steps", "value": 8},
                {"param": "lr_scheduler_type", "value": "linear"},
                {"param": "optim", "value": "paged_adamw_8bit"},
                {"param": "warmup_ratio", "value": 0.1},
                {"param": "weight_decay", "value": 0.05},
                {"param": "learning_rate", "value": 3e-4},
                {"param": "lora_rank", "value": 8},
                {"param": "learning_rate", "value": 1e-3},
                {"param": "per_device_batch_size", "value": 8},
                {"param": "lora_dropout", "value": 0.0},
            ]
            sched_idx = (exp_num - 1) % len(SCHEDULE)
            change = SCHEDULE[sched_idx]
            proposal_config = copy.deepcopy(current_config)
            proposal_config[change["param"]] = change["value"]
            if change["param"] == "lora_rank":
                proposal_config["lora_alpha"] = change["value"] * 2
            description = f"Schedule: {change['param']}={change['value']}"
            hypothesis = f"Testing {change['param']} at {change['value']}"
        else:
            proposal_config = proposal.get("config", current_config)
            description = proposal.get("description", "No description")
            hypothesis = proposal.get("hypothesis", "No hypothesis")
            print(f"Agent proposed: {description}")

    print(f"Proposal: {description}")
    print(f"Hypothesis: {hypothesis}")

    # Step 2: Run training
    try:
        result = run_training(proposal_config, experiment_id)
        val_loss = result["val_loss"]
        train_loss = result["train_loss"]
        peak_vram = result["peak_vram_gb"]
    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        traceback.print_exc()
        val_loss = None
        train_loss = None
        peak_vram = 0

    # Step 3: Keep or revert
    kept = False
    if val_loss is not None and val_loss < best_val_loss:
        kept = True
        prev_best = best_val_loss
        best_val_loss = val_loss
        best_config = copy.deepcopy(proposal_config)
        current_config = copy.deepcopy(proposal_config)
        print(f"KEPT: val_loss improved {prev_best:.6f} -> {val_loss:.6f}")
    else:
        current_config = copy.deepcopy(best_config)
        if val_loss is not None:
            print(f"REVERTED: val_loss {val_loss:.6f} >= best {best_val_loss:.6f}")
        else:
            print("REVERTED: training failed")

    # Step 4: Record
    history_lines.append(f"{experiment_id}\t{description}\t{val_loss or 'FAILED'}\t{kept}")

    with open(RESULTS_FILE, "a") as f:
        f.write(f"{timestamp}\t{experiment_id}\t{description}\t{hypothesis}\t{val_loss or ''}\t{train_loss or ''}\t{peak_vram}\t{kept}\n")

print(f"\n{'='*60}")
print(f"AUTO-RESEARCH COMPLETE")
print(f"Experiments: {MAX_EXPERIMENTS}")
print(f"Best val_loss: {best_val_loss}")
print(f"Best config: {json.dumps(best_config, indent=2)}")
print(f"Results: {RESULTS_FILE}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Final Results

# COMMAND ----------

final = {
    "best_val_loss": best_val_loss,
    "best_config": best_config,
    "total_experiments": MAX_EXPERIMENTS,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

# Save to local tmp first, then copy to UC Volumes
with open("/tmp/auto_research_final.json", "w") as f:
    json.dump(final, f, indent=2)

import shutil
shutil.copy(RESULTS_FILE, UC_RESULTS_FILE)
shutil.copy("/tmp/auto_research_final.json", "/Volumes/main/auto_research/autoresearch/auto_research_final.json")
print("Results copied to UC Volumes")

# Print results table
with open(RESULTS_FILE) as f:
    print(f.read())

dbutils.notebook.exit(json.dumps({"best_val_loss": best_val_loss, "experiments": MAX_EXPERIMENTS}))
