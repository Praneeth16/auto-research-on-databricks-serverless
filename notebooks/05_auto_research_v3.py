# Databricks notebook source
# MAGIC %md
# MAGIC # Auto-Research v3: Karpathy-Style Source Code Editing
# MAGIC
# MAGIC Key changes over v2:
# MAGIC - Agent edits the FULL train.py source code, not just config values
# MAGIC - Fast-fail on NaN / exploding loss (loss > 100 => exit immediately)
# MAGIC - Warmup steps excluded from time budget (first 10 steps)
# MAGIC - Do-not-repeat ledger prevents duplicate failed experiments
# MAGIC - Temperature 0.1 for deterministic code edits
# MAGIC - Dynamic experiment budget with early stopping

# COMMAND ----------

import os, sys, time, json, copy, random, subprocess, shutil, re, math
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
RESULTS_FILE = "/tmp/auto_research_v3_results.tsv"
DO_NOT_REPEAT_FILE = "/tmp/do_not_repeat.md"
BASE_MODEL = "Qwen/Qwen2.5-3B"
TRAIN_SCRIPT_PATH = "/tmp/train_experiment.py"

BASE_EXPERIMENTS = 15
PATIENCE = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base train.py template

# COMMAND ----------

BASE_TRAIN_PY = '''import os, sys, time, math, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/main/auto_research/autoresearch/data/financial_instruct"
TIME_BUDGET = 300  # seconds, DO NOT CHANGE
BASE_MODEL = "Qwen/Qwen2.5-3B"  # DO NOT CHANGE

# --- CONFIGURABLE (agent can modify anything below) ---
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
OPTIM = "adamw_torch"

ds = load_from_disk(DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True)
model.gradient_checkpointing_enable()

model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
    task_type=TaskType.CAUSAL_LM, bias="none"))

class TimeoutCallback(TrainerCallback):
    def __init__(self, budget):
        self.budget = budget; self.t0 = None; self.step = 0
    def on_train_begin(self, *a, **kw): self.t0 = time.time(); self.step = 0
    def on_step_end(self, args, state, control, **kw):
        self.step += 1
        # Exclude first 10 steps from budget (warmup/compile)
        if self.step > 10 and self.t0 and (time.time() - self.t0) > self.budget:
            control.should_training_stop = True
    def on_log(self, args, state, control, logs=None, **kw):
        # Fast-fail on NaN or exploding loss
        if logs and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or loss > 100:
                print("FAIL: loss=" + str(loss))
                control.should_training_stop = True

trainer = SFTTrainer(model=model, processing_class=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    args=SFTConfig(output_dir="/tmp/train_out", per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER, warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY, num_train_epochs=1, optim=OPTIM,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="no", max_length=MAX_LENGTH, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none"))
trainer.add_callback(TimeoutCallback(TIME_BUDGET))

result = trainer.train()
eval_result = trainer.evaluate()
val_loss = eval_result["eval_loss"]
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print("---")
print("val_loss: " + str(val_loss))
print("peak_vram_gb: " + str(round(peak_vram, 2)))
'''

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

TASK_COLUMN = None
MULTI_TASK_MODE = False
task_dist = {}
num_task_types = 1

if TASK_COLUMN is None:
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
        candidates.sort(key=lambda x: -x[1])
        TASK_COLUMN = candidates[0][0]
        print("Auto-detected task column: '" + TASK_COLUMN + "' (" + str(candidates[0][1]) + " unique values)")
        print("  Sample values: " + str(candidates[0][2]))
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
        num_task_types = len([t for t in task_dist if task_dist[t] > 0.01])
        print("MULTI-TASK MODE: " + str(num_task_types) + " tasks via column '" + TASK_COLUMN + "'")
        print("Distribution: " + json.dumps(task_dist))
    else:
        print("SINGLE-TASK MODE: column '" + TASK_COLUMN + "' has only 1 value")
else:
    print("SINGLE-TASK MODE: using val_loss as optimization metric")

data_factor = 1.0 if len(train_ds) < 50000 else 1.5
MAX_EXPERIMENTS = int(BASE_EXPERIMENTS * num_task_types * data_factor)
print("Dynamic budget: " + str(MAX_EXPERIMENTS) + " experiments (patience=" + str(PATIENCE) + ")")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Do-not-repeat ledger

# COMMAND ----------

do_not_repeat = []

with open(DO_NOT_REPEAT_FILE, "w") as f:
    f.write("# Do-Not-Repeat Ledger\n")
    f.write("# Failed experiments that should not be retried.\n\n")


def add_to_do_not_repeat(description, val_loss, reason):
    entry = "- " + description + " | val_loss=" + str(val_loss) + " | reason=" + reason
    do_not_repeat.append(entry)
    with open(DO_NOT_REPEAT_FILE, "w") as f:
        f.write("# Do-Not-Repeat Ledger\n\n")
        for item in do_not_repeat:
            f.write(item + "\n")


def get_do_not_repeat_text():
    if not do_not_repeat:
        return "No failed experiments yet."
    return "\n".join(do_not_repeat)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run train.py as subprocess and parse results

# COMMAND ----------

def run_train_inline(train_py_code, experiment_id):
    """Execute train.py code inline via exec(), capturing val_loss from a results dict.
    This avoids the subprocess MLflow auth issue on Databricks."""
    import io, contextlib

    # Save the code for reference
    with open(TRAIN_SCRIPT_PATH, "w") as f:
        f.write(train_py_code)

    # Patch the code: replace the final print block with writing to a shared dict
    # The code prints "---\nval_loss: X\npeak_vram_gb: Y" at the end.
    # We intercept stdout to capture these values.
    val_loss = None
    peak_vram = None
    stdout_capture = io.StringIO()

    # Set sys.argv for the script
    old_argv = sys.argv
    sys.argv = [TRAIN_SCRIPT_PATH, DATA_PATH]

    try:
        # Redirect stdout to capture val_loss output
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(train_py_code, TRAIN_SCRIPT_PATH, "exec"), {"__name__": "__main__"})
        returncode = 0
    except SystemExit as e:
        returncode = e.code if e.code else 0
    except Exception as e:
        import traceback
        print("Exec error: " + str(e))
        traceback.print_exc()
        returncode = 1
    finally:
        sys.argv = old_argv
        # Always clean up GPU memory
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except: pass

    stdout = stdout_capture.getvalue()
    # Print tail for visibility
    stdout_lines = stdout.strip().split("\n")
    tail = stdout_lines[-20:] if len(stdout_lines) > 20 else stdout_lines
    print("--- exec stdout (tail) ---")
    for line in tail:
        print("  " + line)

    # Parse val_loss and peak_vram from stdout after "---" marker
    found_marker = False
    for line in stdout_lines:
        if line.strip() == "---":
            found_marker = True
            continue
        if found_marker:
            if line.startswith("val_loss: "):
                try:
                    val_loss = float(line.split("val_loss: ")[1].strip())
                except:
                    pass
            if line.startswith("peak_vram_gb: "):
                try:
                    peak_vram = float(line.split("peak_vram_gb: ")[1].strip())
                except:
                    pass

    # Check for FAIL marker (NaN or exploding loss)
    if "FAIL: loss=" in stdout:
        if val_loss is not None and (math.isnan(val_loss) or val_loss > 100):
            val_loss = None  # treat as failed

    return {
        "val_loss": val_loss,
        "peak_vram_gb": peak_vram,
        "stdout": stdout,
        "stderr": "",
        "returncode": returncode,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent: GPT 5.4 with source code editing

# COMMAND ----------

PROGRAM_MD = """You are an ML researcher running experiments to minimize val_loss on a QLoRA fine-tuning task.

## Setup
- GPU: A10G 24GB (g5.xlarge)
- Base model: Qwen/Qwen2.5-3B with QLoRA (4-bit)
- Data: financial instruction-following dataset
- Time budget: 300 seconds per run (enforced by TimeoutCallback)

## Rules
1. Your goal: get the LOWEST val_loss possible.
2. You will see the current train.py source code. Return the FULL modified train.py.
3. You may modify train.py freely: hyperparameters, LoRA config, optimizer, scheduler, training loop, anything.
4. You may NOT change: DATA_PATH, TIME_BUDGET, BASE_MODEL, report_to="none", or the final print block (after "---").
5. Make ONE meaningful change per experiment for clear attribution.
6. Read the experiment history and do-not-repeat list before proposing.
7. The code must run in a single Python file with no extra dependencies beyond what is already imported.
8. The A10G has 24GB VRAM. Do not exceed it (e.g., batch_size=8 with max_length=2048 will OOM).
9. Return ONLY the complete train.py source code, wrapped in ```python ... ``` markers.
10. Before the code block, write ONE line describing what you changed and why.
"""


def call_agent(current_train_py, history_str, do_not_repeat_str, task_info_str):
    """Call GPT 5.4 to propose a modified train.py."""
    system = PROGRAM_MD

    if MULTI_TASK_MODE:
        system = system + "\n\n## Task info\n" + task_info_str
    else:
        system = system + "\n\nThis is a single-task dataset. Optimize val_loss (lower is better)."

    user_parts = [
        "## Current train.py\n```python\n" + current_train_py + "\n```",
        "",
        "## Experiment history",
        history_str if history_str else "No experiments yet.",
        "",
        "## Do-not-repeat list (failed experiments, do NOT retry these)",
        do_not_repeat_str,
        "",
        "Return your ONE-line description, then the FULL modified train.py in ```python ... ``` markers.",
    ]
    user = "\n".join(user_parts)

    try:
        api_url = DB_HOST + "/serving-endpoints/" + AGENT_MODEL + "/invocations"
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        headers = {"Authorization": "Bearer " + DB_TOKEN, "Content-Type": "application/json"}
        r = http_lib.post(api_url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]

        # Extract description (first non-empty line before code block)
        description = ""
        lines_before_code = text.split("```python")[0].strip().split("\n")
        for line in lines_before_code:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped
                break
        if not description:
            description = "Agent-proposed modification"

        # Extract code block
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            code = text.split("```")[1].split("```")[0].strip()
        else:
            print("Agent did not return code block. Raw response (first 500 chars):")
            print(text[:500])
            return None

        # Validate the code has required invariants
        if "DO NOT CHANGE" not in code:
            print("WARNING: agent removed DO NOT CHANGE markers, likely modified invariants")
        if 'print("val_loss: "' not in code and "print('val_loss: '" not in code:
            print("WARNING: agent removed val_loss print statement")
            return None

        return {"code": code, "description": description}

    except Exception as e:
        import traceback
        print("Agent error: " + str(e))
        traceback.print_exc()
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preflight: test agent connectivity

# COMMAND ----------

print("Testing " + AGENT_MODEL + " agent...")
test_payload = {
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10,
    "temperature": 0.1,
}
test_url = DB_HOST + "/serving-endpoints/" + AGENT_MODEL + "/invocations"
test_headers = {"Authorization": "Bearer " + DB_TOKEN, "Content-Type": "application/json"}
test_r = http_lib.post(test_url, json=test_payload, headers=test_headers, timeout=30)
if test_r.status_code != 200:
    raise RuntimeError("Agent test failed. status=" + str(test_r.status_code) + " body=" + test_r.text[:300])
print("Agent OK: " + test_r.json()["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the experiment loop

# COMMAND ----------

best_val_loss = float("inf")
best_train_py = BASE_TRAIN_PY
current_train_py = BASE_TRAIN_PY
history_lines = []
no_improve_count = 0

task_info_str = ""
if MULTI_TASK_MODE:
    task_info_str = "Multi-task dataset. Distribution: " + json.dumps(task_dist)

with open(RESULTS_FILE, "w") as f:
    f.write("timestamp\texperiment_id\tdescription\tval_loss\tpeak_vram_gb\tkept\n")

print("=" * 60)
print("AUTO-RESEARCH v3: " + str(MAX_EXPERIMENTS) + " max experiments, patience=" + str(PATIENCE))
print("Agent: " + AGENT_MODEL + " (temp=0.1)")
print("Mode: source code editing (Karpathy-style)")
print("=" * 60)

for exp_num in range(MAX_EXPERIMENTS):
    experiment_id = "v3_exp_" + str(exp_num).zfill(3)
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 60)
    print("EXPERIMENT " + str(exp_num) + "/" + str(MAX_EXPERIMENTS) + " | best_val_loss=" + str(round(best_val_loss, 6) if best_val_loss < float("inf") else "N/A") + " | no_improve=" + str(no_improve_count) + "/" + str(PATIENCE))

    if no_improve_count >= PATIENCE and exp_num > 0:
        print("EARLY STOP: no improvement in " + str(PATIENCE) + " experiments")
        break

    # --- Baseline or agent proposal ---
    if exp_num == 0:
        proposal_code = current_train_py
        description = "Baseline (unmodified train.py)"
    else:
        history_str = "\n".join(history_lines[-15:])
        do_not_repeat_str = get_do_not_repeat_text()
        proposal = call_agent(current_train_py, history_str, do_not_repeat_str, task_info_str)
        if proposal is None:
            print("Agent failed to produce valid code. Skipping this experiment.")
            no_improve_count += 1
            history_lines.append(experiment_id + " | AGENT_FAILURE | skipped")
            continue
        proposal_code = proposal["code"]
        description = proposal["description"]

    print("Proposal: " + description[:120])

    # --- Run experiment ---
    t0 = time.time()
    result = run_train_inline(proposal_code, experiment_id)
    wall_time = round(time.time() - t0, 1)
    val_loss = result["val_loss"]
    peak_vram = result["peak_vram_gb"]

    print("Result: val_loss=" + str(val_loss) + ", peak_vram=" + str(peak_vram) + ", wall_time=" + str(wall_time) + "s, rc=" + str(result["returncode"]))

    # --- Keep or revert ---
    kept = False
    if val_loss is not None and val_loss < best_val_loss:
        kept = True
        best_val_loss = val_loss
        best_train_py = proposal_code
        current_train_py = proposal_code
        no_improve_count = 0
        print("KEPT: val_loss " + str(round(val_loss, 6)) + " (new best)")

        # Save adapter
        adapter_path = ADAPTER_BASE + "/v3_" + experiment_id
        src_adapter = "/tmp/train_out"
        if os.path.exists(src_adapter):
            try:
                shutil.copytree(src_adapter, adapter_path, dirs_exist_ok=True)
                print("Adapter saved to " + adapter_path)
            except Exception as e:
                print("Adapter save warning: " + str(e))
    else:
        current_train_py = best_train_py
        no_improve_count += 1
        if val_loss is None:
            reason = "crashed (rc=" + str(result["returncode"]) + ")"
            add_to_do_not_repeat(description, "N/A", reason)
            # Save first crash's stderr for debugging
            stderr_text = result.get("stderr", "")
            if stderr_text and exp_num <= 1:
                crash_path = "/tmp/v3_crash_" + experiment_id + ".txt"
                with open(crash_path, "w") as cf:
                    cf.write("STDOUT:\n" + result.get("stdout", "") + "\n\nSTDERR:\n" + stderr_text)
                import shutil
                try:
                    shutil.copy(crash_path, "/Volumes/main/auto_research/autoresearch/v3_crash_" + experiment_id + ".txt")
                except: pass
            print("REVERTED: experiment crashed")
        else:
            reason = "val_loss " + str(round(val_loss, 6)) + " >= best " + str(round(best_val_loss, 6))
            add_to_do_not_repeat(description, str(round(val_loss, 6)), reason)
            print("REVERTED: " + reason)

    # --- Record ---
    history_entry = experiment_id + " | " + description + " | val_loss=" + str(val_loss) + " | kept=" + str(kept)
    history_lines.append(history_entry)
    with open(RESULTS_FILE, "a") as f:
        row = "\t".join([
            timestamp, experiment_id, description,
            str(val_loss if val_loss is not None else ""),
            str(peak_vram if peak_vram is not None else ""),
            str(kept),
        ])
        f.write(row + "\n")

print("\n" + "=" * 60)
print("AUTO-RESEARCH v3 COMPLETE")
print("Experiments: " + str(exp_num + 1))
print("Best val_loss: " + str(round(best_val_loss, 6) if best_val_loss < float("inf") else "N/A"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results and best train.py

# COMMAND ----------

final = {
    "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
    "total_experiments": exp_num + 1,
    "early_stopped": no_improve_count >= PATIENCE,
    "agent_model": AGENT_MODEL,
    "agent_temperature": 0.1,
    "mode": "source_code_editing",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open("/tmp/auto_research_v3_final.json", "w") as f:
    json.dump(final, f, indent=2)

with open("/tmp/auto_research_v3_best_train.py", "w") as f:
    f.write(best_train_py)

# Copy to UC Volumes
shutil.copy(RESULTS_FILE, "/Volumes/main/auto_research/autoresearch/auto_research_v3_results.tsv")
shutil.copy("/tmp/auto_research_v3_final.json", "/Volumes/main/auto_research/autoresearch/auto_research_v3_final.json")
shutil.copy("/tmp/auto_research_v3_best_train.py", "/Volumes/main/auto_research/autoresearch/auto_research_v3_best_train.py")
shutil.copy(DO_NOT_REPEAT_FILE, "/Volumes/main/auto_research/autoresearch/auto_research_v3_do_not_repeat.md")

print("Results TSV:")
with open(RESULTS_FILE) as f:
    print(f.read())

print("\nDo-not-repeat ledger:")
with open(DO_NOT_REPEAT_FILE) as f:
    print(f.read())

print("\nBest train.py (first 50 lines):")
best_lines = best_train_py.strip().split("\n")[:50]
for line in best_lines:
    print("  " + line)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exit

# COMMAND ----------

exit_data = {"best_val_loss": best_val_loss if best_val_loss < float("inf") else None, "experiments": exp_num + 1, "early_stopped": no_improve_count >= PATIENCE}
dbutils.notebook.exit(json.dumps(exit_data))
