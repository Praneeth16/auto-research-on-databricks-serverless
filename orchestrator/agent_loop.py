"""
Core auto-research loop: call agent LLM, parse diff, submit training job, keep/revert.

This is the heart of the system. It runs on serverless CPU compute and orchestrates
GPU training experiments autonomously.
"""

import os
import csv
import time
import json
import shutil
import difflib
from pathlib import Path
from datetime import datetime, timezone

import mlflow
from databricks.sdk import WorkspaceClient
from openai import OpenAI

from orchestrator.config import AutoResearchConfig
from orchestrator.submit_training import submit_training_run, setup_uc_volumes


def load_program_md() -> str:
    """Load the agent instructions from program.md."""
    program_path = Path("program.md")
    if not program_path.exists():
        raise FileNotFoundError("program.md not found in project root")
    return program_path.read_text()


def load_current_train_py() -> str:
    """Load the current train.py that the agent will modify."""
    train_path = Path("train.py")
    if not train_path.exists():
        raise FileNotFoundError("train.py not found in project root")
    return train_path.read_text()


def load_results_history(config: AutoResearchConfig) -> str:
    """Load the experiment history from results.tsv."""
    results_path = Path(config.local_results)
    if not results_path.exists():
        return "No experiments run yet."

    content = results_path.read_text().strip()
    if not content:
        return "No experiments run yet."

    lines = content.split("\n")
    # Show last 20 experiments for context
    if len(lines) > 21:  # header + 20 rows
        return "\n".join([lines[0]] + lines[-20:])
    return content


def load_notes(config: AutoResearchConfig) -> str:
    """Load the agent's notes from previous experiments."""
    notes_path = Path(config.local_notes)
    if notes_path.exists():
        return notes_path.read_text().strip() or "No notes yet."
    return "No notes yet."


def compute_diff(old_code: str, new_code: str) -> str:
    """Compute a unified diff between old and new train.py."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="train.py.old", tofile="train.py.new")
    return "".join(diff)


def call_agent_llm(
    config: AutoResearchConfig,
    program: str,
    current_code: str,
    history: str,
    notes: str,
) -> dict:
    """
    Call the agent LLM to propose a modification to train.py.

    Returns dict with:
        - new_code: the modified train.py content
        - description: what the agent changed and why
        - hypothesis: what the agent expects to happen
    """
    # Use Databricks Foundation Model API via OpenAI-compatible endpoint
    client = OpenAI(
        api_key=os.environ.get("DATABRICKS_TOKEN", ""),
        base_url=f"{config.host}/serving-endpoints",
    )

    system_prompt = """You are an autonomous ML researcher. Your job is to improve a LoRA fine-tuning
script by making ONE targeted change per experiment. You will receive:
1. Research instructions (program.md)
2. The current train.py
3. History of past experiments and their results
4. Notes from previous experiments

You must respond with a JSON object containing:
- "new_code": the complete modified train.py (not a diff, the FULL file)
- "description": a one-line description of what you changed
- "hypothesis": what you expect this change to do to val_loss

Make exactly ONE change. Do not modify the MLflow logging, the val_loss output line,
or the argument parsing. Focus on LoRA config, training hyperparameters, data preprocessing,
or optimization settings."""

    user_prompt = f"""## Research Instructions
{program}

## Current train.py
```python
{current_code}
```

## Experiment History
{history}

## Notes from Previous Experiments
{notes}

Based on the history and instructions, propose your next experiment.
Respond with a JSON object containing "new_code", "description", and "hypothesis"."""

    response = client.chat.completions.create(
        model=config.agent_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8192,
        temperature=0.7,
    )

    response_text = response.choices[0].message.content

    # Parse JSON from response (handle markdown code blocks)
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
        else:
            raise ValueError(f"Could not parse agent response as JSON: {response_text[:500]}")

    return {
        "new_code": result["new_code"],
        "description": result.get("description", "No description provided"),
        "hypothesis": result.get("hypothesis", "No hypothesis provided"),
    }


def append_result(config: AutoResearchConfig, row: dict):
    """Append a result row to results.tsv."""
    results_path = Path(config.local_results)
    file_exists = results_path.exists() and results_path.stat().st_size > 0

    fieldnames = [
        "timestamp", "experiment_id", "description", "hypothesis",
        "val_loss", "prev_val_loss", "improved", "kept",
        "duration_seconds", "run_id", "status",
    ]

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_notes(config: AutoResearchConfig, note: str):
    """Append a note to the agent's notes file."""
    notes_path = Path(config.local_notes)
    with open(notes_path, "a") as f:
        f.write(f"\n{note}\n")


def run_auto_research(config: AutoResearchConfig = None):
    """
    Main auto-research loop.

    1. Load program.md and current train.py
    2. Call agent LLM to propose a change
    3. Submit training run on Databricks GPU
    4. Compare val_loss with previous best
    5. Keep or revert the change
    6. Log everything to MLflow
    7. Repeat
    """
    if config is None:
        config = AutoResearchConfig()

    # Initialize Databricks client
    client = WorkspaceClient(
        host=config.host,
        profile=config.profile,
    )

    # Set up UC Volumes
    setup_uc_volumes(client, config)

    # Set up MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(config.experiment_name)

    # Load initial state
    program = load_program_md()
    best_val_loss = float("inf")
    best_code = load_current_train_py()
    experiment_count = 0

    # Check if resuming from previous run
    results_path = Path(config.local_results)
    if results_path.exists():
        with open(results_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                experiment_count += 1
                if row.get("kept") == "True" and row.get("val_loss"):
                    try:
                        best_val_loss = float(row["val_loss"])
                    except ValueError:
                        pass
        print(f"Resuming from experiment {experiment_count}, best val_loss: {best_val_loss}")

    print(f"\n{'='*60}")
    print(f"AUTO-RESEARCH LOOP STARTING")
    print(f"Max experiments: {config.max_experiments}")
    print(f"Training budget: {config.max_training_seconds}s per experiment")
    print(f"Agent LLM: {config.agent_model}")
    print(f"GPU: {config.cluster.node_type_id}")
    print(f"Best val_loss so far: {best_val_loss}")
    print(f"{'='*60}\n")

    while experiment_count < config.max_experiments:
        experiment_id = f"exp_{experiment_count:04d}"
        timestamp = datetime.now(timezone.utc).isoformat()

        print(f"\n--- Experiment {experiment_id} ---")

        # Step 1: Call agent LLM
        print("Calling agent LLM for next experiment proposal...")
        current_code = load_current_train_py()
        history = load_results_history(config)
        notes = load_notes(config)

        try:
            proposal = call_agent_llm(config, program, current_code, history, notes)
        except Exception as e:
            print(f"Agent LLM failed: {e}")
            append_notes(config, f"[{experiment_id}] Agent LLM error: {e}")
            experiment_count += 1
            continue

        new_code = proposal["new_code"]
        description = proposal["description"]
        hypothesis = proposal["hypothesis"]

        print(f"Proposal: {description}")
        print(f"Hypothesis: {hypothesis}")

        # Step 2: Write new train.py
        diff = compute_diff(current_code, new_code)
        print(f"Diff size: {len(diff)} chars")

        train_path = Path(config.local_train_script)
        train_path.write_text(new_code)

        # Step 3: Submit training run
        print(f"Submitting training run on {config.cluster.node_type_id}...")
        try:
            result = submit_training_run(client, config, experiment_id, description)
        except Exception as e:
            print(f"Training submission failed: {e}")
            # Revert train.py
            train_path.write_text(current_code)
            append_notes(config, f"[{experiment_id}] Submission failed: {e}")
            append_result(config, {
                "timestamp": timestamp,
                "experiment_id": experiment_id,
                "description": description,
                "hypothesis": hypothesis,
                "val_loss": "",
                "prev_val_loss": best_val_loss if best_val_loss != float("inf") else "",
                "improved": "False",
                "kept": "False",
                "duration_seconds": 0,
                "run_id": "",
                "status": "SUBMISSION_FAILED",
            })
            experiment_count += 1
            continue

        val_loss = result["val_loss"]
        run_id = result["run_id"]
        duration = result["duration_seconds"]
        status = result["status"]

        print(f"Run {run_id} completed: status={status}, val_loss={val_loss}, duration={duration}s")

        # Step 4: Keep or revert
        improved = False
        kept = False
        prev_best = best_val_loss  # capture before any update

        if val_loss is not None and val_loss < best_val_loss:
            improved = True
            kept = True
            best_val_loss = val_loss
            best_code = new_code

            # Save best to live/
            live_path = Path(config.local_live_dir) / "train.py"
            live_path.write_text(new_code)

            print(f"KEPT: val_loss improved {prev_best:.6f} -> {val_loss:.6f}")
        else:
            # Revert
            train_path.write_text(best_code if best_val_loss != float("inf") else current_code)
            reason = "no val_loss returned" if val_loss is None else f"val_loss {val_loss:.6f} >= best {best_val_loss:.6f}"
            print(f"REVERTED: {reason}")

        # Step 5: Log to MLflow
        with mlflow.start_run(run_name=experiment_id):
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("description", description)
            mlflow.log_param("hypothesis", hypothesis)
            mlflow.log_param("kept", kept)

            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("duration_seconds", duration)
            if best_val_loss != float("inf"):
                mlflow.log_metric("best_val_loss", best_val_loss)

            mlflow.log_text(new_code, "train.py")
            mlflow.log_text(diff, "diff.patch")

            mlflow.set_tag("status", status)
            mlflow.set_tag("improved", str(improved))
            mlflow.set_tag("databricks_run_id", str(run_id))

        # Step 6: Record result
        append_result(config, {
            "timestamp": timestamp,
            "experiment_id": experiment_id,
            "description": description,
            "hypothesis": hypothesis,
            "val_loss": "" if val_loss is None else val_loss,
            "prev_val_loss": "" if prev_best == float("inf") else prev_best,
            "improved": str(improved),
            "kept": str(kept),
            "duration_seconds": duration,
            "run_id": run_id,
            "status": status,
        })

        # Add notes for failed experiments
        if not kept:
            note = f"[{experiment_id}] {description} -> val_loss={val_loss}, not kept. {hypothesis}"
            append_notes(config, note)

        experiment_count += 1

    print(f"\n{'='*60}")
    print(f"AUTO-RESEARCH COMPLETE")
    print(f"Total experiments: {experiment_count}")
    print(f"Best val_loss: {best_val_loss}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_auto_research()
