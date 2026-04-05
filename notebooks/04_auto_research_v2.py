# Databricks notebook source
# MAGIC %md
# MAGIC # Auto-Research v2: Multi-Task Aware, GPT 5.4 Agent
# MAGIC
# MAGIC Improvements over v1:
# MAGIC - Per-task evaluation (sentiment, Q&A, extraction) after each experiment
# MAGIC - GPT 5.4 as agent (better reasoning, reliable JSON)
# MAGIC - Agent sees data distribution and per-task metrics
# MAGIC - Dynamic experiment budget with convergence detection

# COMMAND ----------

import os, sys, time, json, copy, random
import torch
import numpy as np
from datetime import datetime, timezone
from collections import Counter

import requests as http_lib
DB_HOST = spark.conf.get("spark.databricks.workspaceUrl", "")
if not DB_HOST.startswith("http"): DB_HOST = "https://" + DB_HOST
DB_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
print("FM API: host=" + DB_HOST + ", token=" + ("present" if DB_TOKEN else "MISSING"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

AGENT_MODEL = "databricks-gpt-5-4"
TRAINING_BUDGET_SECONDS = 300
DATA_PATH = "/Volumes/main/auto_research/autoresearch/data/financial_instruct"
ADAPTER_BASE = "/Volumes/main/auto_research/autoresearch/adapters"
RESULTS_FILE = "/tmp/auto_research_v2_results.tsv"
BASE_MODEL = "Qwen/Qwen2.5-3B"

# Dynamic budget
BASE_EXPERIMENTS = 15
PATIENCE = 5
MIN_DELTA = 0.005

# Task weights: auto-detected from data if task_type column exists.
# Override here if you want specific weights; otherwise computed equally.
TASK_WEIGHTS_OVERRIDE = None  # Set to a dict to override, e.g. {"sentiment": 0.5, "qa": 0.3}


DEFAULT_CONFIG = {
    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4, "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03, "weight_decay": 0.01,
    "per_device_batch_size": 4, "gradient_accumulation_steps": 4,
    "max_length": 1024, "gradient_checkpointing": True,
    "bf16": True, "optim": "adamw_torch",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and compute dynamic budget

# COMMAND ----------

from datasets import load_from_disk

ds = load_from_disk(DATA_PATH)
train_ds = ds["train"]
val_ds = ds["validation"]
print("Train: " + str(len(train_ds)) + ", Val: " + str(len(val_ds)))
print("Columns: " + str(val_ds.column_names))

# --- Auto-detect task column ---
# Users can set TASK_COLUMN explicitly. If not set, the skill looks for
# candidate columns (categorical with 2-20 unique values, not "messages").
# If no candidate is found, runs in single-task mode.

TASK_COLUMN = None  # <-- SET THIS if your column isn't auto-detected, e.g. "task_type", "category", "label"

MULTI_TASK_MODE = False
task_dist = {}
eval_by_task = {}
TASK_WEIGHTS = {}
EVAL_PER_TASK = 25

if TASK_COLUMN is None:
    # Auto-detect: find columns with 2-20 unique string values (likely task/category columns)
    candidates = []
    for col in val_ds.column_names:
        if col == "messages":
            continue
        try:
            sample = val_ds[col][:100]
            if all(isinstance(v, str) for v in sample):
                unique_vals = set(val_ds[col])
                if 2 <= len(unique_vals) <= 20:
                    candidates.append((col, len(unique_vals), list(unique_vals)[:5]))
        except:
            pass

    if candidates:
        # Pick the one with the most unique values (most likely to be task type)
        candidates.sort(key=lambda x: -x[1])
        TASK_COLUMN = candidates[0][0]
        print("Auto-detected task column: '" + TASK_COLUMN + "' (" + str(candidates[0][1]) + " unique values)")
        print("  Sample values: " + str(candidates[0][2]))
        if len(candidates) > 1:
            print("  Other candidates: " + str([(c[0], c[1]) for c in candidates[1:]]))
            print("  Set TASK_COLUMN explicitly if this is wrong.")
    else:
        print("No task column detected. Running in SINGLE-TASK mode.")

if TASK_COLUMN and TASK_COLUMN in val_ds.column_names:
    task_counts = Counter(val_ds[TASK_COLUMN])
    task_counts = {k: v for k, v in task_counts.items() if k and str(k).strip()}
    total_val = sum(task_counts.values())

    if len(task_counts) > 1:
        MULTI_TASK_MODE = True
        for k, v in sorted(task_counts.items(), key=lambda x: -x[1]):
            task_dist[k] = round(v / total_val, 3)

        print("MULTI-TASK MODE: " + str(len(task_dist)) + " tasks via column '" + TASK_COLUMN + "'")
        print("Distribution: " + json.dumps(task_dist))

        # Auto-compute weights: inverse proportion (rare tasks weighted higher)
        if TASK_WEIGHTS_OVERRIDE:
            TASK_WEIGHTS = TASK_WEIGHTS_OVERRIDE
        else:
            inv_props = {k: 1.0 / max(v, 0.01) for k, v in task_dist.items()}
            total_inv = sum(inv_props.values())
            TASK_WEIGHTS = {k: round(v / total_inv, 3) for k, v in inv_props.items()}
        print("Task weights: " + json.dumps(TASK_WEIGHTS))

        # Per-task eval samples
        for task_type in task_dist:
            examples = [row for row in val_ds if row[TASK_COLUMN] == task_type]
            random.seed(42)
            if len(examples) > EVAL_PER_TASK:
                examples = random.sample(examples, EVAL_PER_TASK)
            eval_by_task[task_type] = examples
            print("  " + task_type + ": " + str(len(examples)) + " eval samples")
    else:
        print("SINGLE-TASK MODE: column '" + TASK_COLUMN + "' has only 1 value")
else:
    print("SINGLE-TASK MODE: using val_loss as optimization metric")

# Dynamic experiment budget
if MULTI_TASK_MODE:
    num_task_types = len([t for t in task_dist if task_dist[t] > 0.01])
else:
    num_task_types = 1
data_factor = 1.0 if len(train_ds) < 50000 else 1.5
MAX_EXPERIMENTS = int(BASE_EXPERIMENTS * num_task_types * data_factor)
print("Dynamic budget: " + str(MAX_EXPERIMENTS) + " experiments (patience=" + str(PATIENCE) + ")")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-task evaluation

# COMMAND ----------

def evaluate_per_task(model, tokenizer, eval_by_task):
    model.eval()
    results = {}
    for task_type, examples in eval_by_task.items():
        if not examples:
            continue
        correct = 0
        total_count = len(examples)
        for ex in examples:
            msgs = ex["messages"]
            user_msg = msgs[1]["content"]
            expected = msgs[2]["content"].strip().lower()
            prompt = msgs[0]["content"] + "\n\nUser: " + user_msg + "\n\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            if task_type == "sentiment_classification":
                pred = "neutral"
                for label in ["positive", "negative", "neutral"]:
                    if label in response:
                        pred = label
                        break
                if pred == expected:
                    correct += 1
            else:
                expected_words = set(expected.split())
                response_words = set(response.split())
                if len(expected_words) > 0:
                    overlap = len(expected_words & response_words) / len(expected_words)
                    if overlap > 0.3:
                        correct += 1
        acc = correct / total_count if total_count > 0 else 0
        results[task_type] = {"accuracy": round(acc, 4), "total": total_count, "correct": correct}

    composite = 0
    for task_type, metrics in results.items():
        weight = TASK_WEIGHTS.get(task_type, 0.1)
        composite += weight * metrics["accuracy"]
    results["_composite_score"] = round(composite, 4)
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training function

# COMMAND ----------

def run_training(config, experiment_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    print("\n--- Training " + experiment_id + " ---")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto", trust_remote_code=True,
    )
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    class Timeout(TrainerCallback):
        def __init__(self, s): self.s = s; self.t0 = None
        def on_train_begin(self, *a, **kw): self.t0 = time.time()
        def on_step_end(self, args, state, control, **kw):
            if self.t0 and (time.time() - self.t0) > self.s: control.should_training_stop = True

    training_args = SFTConfig(
        output_dir="/tmp/autoresearch_v2/" + experiment_id,
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"], weight_decay=config["weight_decay"],
        num_train_epochs=1, optim=config["optim"], bf16=config["bf16"],
        logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="no",
        max_length=config["max_length"], gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config["gradient_checkpointing"] else None,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds, args=training_args,
    )
    trainer.add_callback(Timeout(TRAINING_BUDGET_SECONDS))

    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    eval_result = trainer.evaluate()
    val_loss = eval_result["eval_loss"]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print("Running per-task evaluation...")
    task_metrics = evaluate_per_task(model, tokenizer, eval_by_task)
    composite = task_metrics["_composite_score"]
    for t, m in task_metrics.items():
        if t != "_composite_score":
            print("  " + t + ": " + str(m["accuracy"]))

    adapter_path = ADAPTER_BASE + "/v2_" + experiment_id
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    del model, trainer
    torch.cuda.empty_cache()

    return {
        "val_loss": val_loss,
        "train_loss": train_result.metrics.get("train_loss", None),
        "train_time_seconds": round(train_time, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "composite_score": composite,
        "task_metrics": task_metrics,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPT 5.4 Agent (data-aware)

# COMMAND ----------

def call_agent(current_config, history, task_dist_str, task_metrics_str):
    system_parts = ["You are an ML researcher optimizing LoRA fine-tuning of a 3B model."]
    if MULTI_TASK_MODE:
        system_parts += [
            "", "MULTI-TASK dataset. Distribution:", task_dist_str,
            "", "Task weights:", json.dumps(TASK_WEIGHTS, indent=2),
            "", "Latest per-task accuracy:", task_metrics_str,
            "", "Goal: maximize weighted composite score. Focus on the weakest task relative to its weight.",
        ]
    else:
        system_parts += [
            "", "Single-task dataset. Goal: minimize val_loss.",
            "", "Latest metrics:", task_metrics_str,
        ]
    system_parts += [
        "", "Propose ONE config change.",
        'Reply ONLY JSON: {"param": "name", "value": new_value, "description": "what and why", "hypothesis": "expected effect"}',
        "", "Params: learning_rate (1e-6 to 1e-3), lora_rank (4-128), lora_alpha (8-256), lora_dropout (0-0.3),",
        "lr_scheduler_type, warmup_ratio (0-0.2), weight_decay (0-0.1), per_device_batch_size (1-8),",
        "gradient_accumulation_steps (1-16), max_length (256-2048), optim (adamw_torch/paged_adamw_8bit)"
    ]
    system = "\n".join(system_parts)

    config_str = "lr=" + str(current_config["learning_rate"]) + ", rank=" + str(current_config["lora_rank"]) + ", alpha=" + str(current_config["lora_alpha"]) + ", dropout=" + str(current_config["lora_dropout"]) + ", max_len=" + str(current_config["max_length"]) + ", optim=" + current_config["optim"] + ", batch=" + str(current_config["per_device_batch_size"]) + ", grad_accum=" + str(current_config["gradient_accumulation_steps"])

    user = "Current config: " + config_str + "\n\nHistory:\n" + history + "\n\nPropose ONE change. ONLY JSON."

    try:
        api_url = DB_HOST + "/serving-endpoints/" + AGENT_MODEL + "/invocations"
        payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 512, "temperature": 0.4}
        headers = {"Authorization": "Bearer " + DB_TOKEN, "Content-Type": "application/json"}
        r = http_lib.post(api_url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        print("Agent: " + text[:200])

        if "```" in text:
            text = text.split("```")[1].split("```")[0]
        if text.strip().startswith("json"):
            text = text.strip()[4:]
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end])

        param = parsed["param"]
        value = parsed["value"]
        new_config = copy.deepcopy(current_config)
        if param in new_config:
            new_config[param] = value
            if param == "lora_rank":
                new_config["lora_alpha"] = value * 2
        return {
            "config": new_config,
            "description": parsed.get("description", "Changed " + param + " to " + str(value)),
            "hypothesis": parsed.get("hypothesis", ""),
        }
    except Exception as e:
        import traceback
        print("Agent error: " + str(e))
        traceback.print_exc()
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preflight: test agent

# COMMAND ----------

print("Testing GPT 5.4 agent...")
test_result = call_agent(DEFAULT_CONFIG, "No experiments yet.", json.dumps(task_dist), "No metrics yet (first run)")
if test_result is None:
    # Diagnostic
    test_url = DB_HOST + "/serving-endpoints/" + AGENT_MODEL + "/invocations"
    test_r = http_lib.post(test_url, json={"messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10},
                           headers={"Authorization": "Bearer " + DB_TOKEN, "Content-Type": "application/json"}, timeout=30)
    raise RuntimeError("Agent failed. Raw status=" + str(test_r.status_code) + " body=" + test_r.text[:300])
print("Agent OK: " + test_result.get("description", "no desc"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the loop

# COMMAND ----------

best_composite = -1.0
best_val_loss = float("inf")
best_config = copy.deepcopy(DEFAULT_CONFIG)
current_config = copy.deepcopy(DEFAULT_CONFIG)
latest_task_metrics = "No metrics yet"
history_lines = ["exp\tdescription\tcomposite\tsentiment_acc\tval_loss\tkept"]
no_improve_count = 0

with open(RESULTS_FILE, "w") as f:
    f.write("timestamp\texperiment_id\tdescription\thypothesis\tval_loss\tcomposite_score\tsentiment_acc\tqa_acc\tkept\n")

print("=" * 60)
print("AUTO-RESEARCH v2: " + str(MAX_EXPERIMENTS) + " max experiments, patience=" + str(PATIENCE))
print("Agent: " + AGENT_MODEL)
print("=" * 60)

# Predefined schedule as fallback
SCHEDULE = [
    {"param": "learning_rate", "value": 5e-4},
    {"param": "learning_rate", "value": 1e-4},
    {"param": "lora_rank", "value": 32},
    {"param": "lora_rank", "value": 64},
    {"param": "max_length", "value": 512},
    {"param": "max_length", "value": 2048},
    {"param": "gradient_accumulation_steps", "value": 8},
    {"param": "lora_dropout", "value": 0.1},
    {"param": "per_device_batch_size", "value": 2},
    {"param": "optim", "value": "paged_adamw_8bit"},
    {"param": "warmup_ratio", "value": 0.1},
    {"param": "weight_decay", "value": 0.05},
    {"param": "learning_rate", "value": 3e-4},
    {"param": "lr_scheduler_type", "value": "linear"},
    {"param": "lora_rank", "value": 8},
]

for exp_num in range(MAX_EXPERIMENTS):
    experiment_id = "v2_exp_" + str(exp_num).zfill(3)
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 60)
    print("EXPERIMENT " + str(exp_num) + "/" + str(MAX_EXPERIMENTS) + " | best_composite=" + str(best_composite) + " | no_improve=" + str(no_improve_count) + "/" + str(PATIENCE))

    # Early stopping check
    if no_improve_count >= PATIENCE and exp_num > 0:
        print("EARLY STOP: no improvement in " + str(PATIENCE) + " experiments")
        break

    # Get proposal
    history_str = "\n".join(history_lines[-15:])
    if exp_num == 0:
        proposal_config = current_config
        description = "Baseline"
        hypothesis = "Establish baseline"
    else:
        proposal = call_agent(current_config, history_str, json.dumps(task_dist), str(latest_task_metrics))
        if proposal is None:
            sched_idx = (exp_num - 1) % len(SCHEDULE)
            change = SCHEDULE[sched_idx]
            proposal_config = copy.deepcopy(current_config)
            proposal_config[change["param"]] = change["value"]
            if change["param"] == "lora_rank":
                proposal_config["lora_alpha"] = change["value"] * 2
            description = "Schedule: " + change["param"] + "=" + str(change["value"])
            hypothesis = "Fallback schedule"
        else:
            proposal_config = proposal["config"]
            description = proposal["description"]
            hypothesis = proposal["hypothesis"]

    print("Proposal: " + description)

    # Train
    try:
        result = run_training(proposal_config, experiment_id)
        val_loss = result["val_loss"]
        composite = result["composite_score"]
        task_metrics = result["task_metrics"]
        sentiment_acc = task_metrics.get("sentiment_classification", {}).get("accuracy", 0)
        qa_acc = task_metrics.get("financial_qa", {}).get("accuracy", 0)
        latest_task_metrics = json.dumps({k: v for k, v in task_metrics.items() if k != "_composite_score"}, indent=2)
    except Exception as e:
        import traceback
        print("Training failed: " + str(e))
        traceback.print_exc()
        val_loss = None
        composite = -1
        sentiment_acc = 0
        qa_acc = 0
        task_metrics = {}

    # Keep or revert
    kept = False
    if MULTI_TASK_MODE:
        # Multi-task: optimize composite score
        improved = composite > best_composite + MIN_DELTA
    else:
        # Single-task: optimize val_loss (lower is better)
        improved = val_loss is not None and val_loss < best_val_loss - MIN_DELTA

    if improved:
        kept = True
        best_composite = composite
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
        best_config = copy.deepcopy(proposal_config)
        current_config = copy.deepcopy(proposal_config)
        no_improve_count = 0
        if MULTI_TASK_MODE:
            print("KEPT: composite " + str(best_composite) + " (sentiment=" + str(sentiment_acc) + ", qa=" + str(qa_acc) + ")")
        else:
            print("KEPT: val_loss " + str(val_loss))
    else:
        current_config = copy.deepcopy(best_config)
        no_improve_count += 1
        print("REVERTED: composite " + str(composite) + " <= best " + str(best_composite))

    # Record
    history_lines.append(experiment_id + "\t" + description + "\t" + str(composite) + "\t" + str(sentiment_acc) + "\t" + str(val_loss or "FAIL") + "\t" + str(kept))
    with open(RESULTS_FILE, "a") as f:
        f.write(timestamp + "\t" + experiment_id + "\t" + description + "\t" + hypothesis + "\t" + str(val_loss or "") + "\t" + str(composite) + "\t" + str(sentiment_acc) + "\t" + str(qa_acc) + "\t" + str(kept) + "\n")

print("\n" + "=" * 60)
print("AUTO-RESEARCH v2 COMPLETE")
print("Experiments: " + str(exp_num + 1))
print("Best composite: " + str(best_composite))
print("Best val_loss: " + str(best_val_loss))
print("Best config: " + json.dumps(best_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

import shutil

final = {
    "best_composite_score": best_composite,
    "best_val_loss": best_val_loss,
    "best_config": best_config,
    "total_experiments": exp_num + 1,
    "early_stopped": no_improve_count >= PATIENCE,
    "latest_task_metrics": task_metrics if task_metrics else {},
    "task_weights": TASK_WEIGHTS,
    "agent_model": AGENT_MODEL,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open("/tmp/auto_research_v2_final.json", "w") as f:
    json.dump(final, f, indent=2)

shutil.copy(RESULTS_FILE, "/Volumes/main/auto_research/autoresearch/auto_research_v2_results.tsv")
shutil.copy("/tmp/auto_research_v2_final.json", "/Volumes/main/auto_research/autoresearch/auto_research_v2_final.json")

with open(RESULTS_FILE) as f:
    print(f.read())

dbutils.notebook.exit(json.dumps({"best_composite": best_composite, "best_val_loss": best_val_loss, "experiments": exp_num + 1, "early_stopped": no_improve_count >= PATIENCE}))
