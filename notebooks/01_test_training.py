# Databricks notebook source
# MAGIC %md
# MAGIC # Test: Single LoRA Fine-Tuning Run
# MAGIC Verify train.py works on a GPU cluster (g5.xlarge / A10G).

# COMMAND ----------

# MAGIC %pip install peft trl bitsandbytes accelerate --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check data

# COMMAND ----------

data_path = "/Volumes/main/auto_research/autoresearch/data/financial_instruct"
display(dbutils.fs.ls(data_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run train.py (2 min budget)

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable,
        "/Volumes/main/auto_research/autoresearch/train.py",
        "--data-path", "/Volumes/main/auto_research/autoresearch/data/financial_instruct",
        "--output-dir", "/Volumes/main/auto_research/autoresearch/adapters",
        "--experiment-id", "test_run_001",
        "--max-seconds", "120",
        "--results-file", "/Volumes/main/auto_research/autoresearch/val_loss_test.txt",
    ],
    capture_output=True,
    text=True,
    timeout=600,
)

print("=== STDOUT ===")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print(f"\n=== RETURN CODE: {result.returncode} ===")
if result.returncode != 0:
    print("=== STDERR ===")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify output

# COMMAND ----------

try:
    print(dbutils.fs.head("/Volumes/main/auto_research/autoresearch/val_loss_test.txt"))
except:
    print("Results file not found. Check STDOUT above for val_loss line.")

try:
    display(dbutils.fs.ls("/Volumes/main/auto_research/autoresearch/adapters/test_run_001"))
except:
    print("Adapter directory not found.")
