# Databricks notebook source
# MAGIC %md
# MAGIC # Quick Eval: LoRA Model vs Sonnet 4.6
# MAGIC Sanity check on 50 examples before launching auto-research loop.

# COMMAND ----------

import os, sys, time, json, torch
import numpy as np
from collections import Counter

# COMMAND ----------

# Load eval examples from our training data's validation set (already on UC Volumes)
from datasets import load_from_disk
full_ds = load_from_disk("/Volumes/main/auto_research/autoresearch/data/financial_instruct")
val_ds = full_ds["validation"]

# Filter to sentiment classification tasks only
import random
random.seed(42)
sentiment_examples = []
for row in val_ds:
    msgs = row["messages"]
    if len(msgs) >= 3 and "sentiment" in msgs[1]["content"].lower():
        text = msgs[1]["content"]
        # Extract the quoted text
        if '"' in text:
            quoted = text.split('"')[1] if '"' in text else text
        else:
            quoted = text.split(":\n\n")[-1] if ":\n\n" in text else text
        label = msgs[2]["content"].strip().lower()
        if label in ("positive", "negative", "neutral"):
            sentiment_examples.append({"text": quoted, "label": label})

examples = random.sample(sentiment_examples, min(50, len(sentiment_examples)))
print(f"Eval: {len(examples)} sentiment examples, dist: {Counter(e['label'] for e in examples)}")

# COMMAND ----------

PROMPT = """Classify the sentiment of this financial text as exactly one of: positive, negative, or neutral.

Text: "{text}"

Sentiment:"""

def parse(resp):
    r = resp.strip().lower()
    for l in ["positive","negative","neutral"]:
        if l in r: return l
    return "neutral"

labels = [e["label"] for e in examples]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. LoRA Model

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "/Volumes/main/auto_research/autoresearch/adapters/v3_v3_exp_017")
model.eval()

lora_preds, lora_lats = [], []
for i, ex in enumerate(examples):
    inp = tok(PROMPT.format(text=ex["text"]), return_tensors="pt", truncation=True, max_length=512).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False)
    lora_lats.append((time.time()-t0)*1000)
    resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    lora_preds.append(parse(resp))

lora_acc = sum(1 for l,p in zip(labels, lora_preds) if l==p) / len(labels)
print(f"LoRA Qwen 2.5-3B: accuracy={lora_acc:.2%}, avg_latency={np.mean(lora_lats):.0f}ms")

del model; torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sonnet 4.6 via FM API

# COMMAND ----------

from openai import OpenAI

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(api_key=db_token, base_url=f"{db_host}/serving-endpoints")

sonnet_preds, sonnet_lats, errors = [], [], 0
for i, ex in enumerate(examples):
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="databricks-claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "Respond with exactly one word: positive, negative, or neutral."},
                {"role": "user", "content": PROMPT.format(text=ex["text"])},
            ],
            max_tokens=5, temperature=0,
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = "neutral"
        errors += 1
        if errors <= 3: print(f"  Error {i}: {e}")
    sonnet_lats.append((time.time()-t0)*1000)
    sonnet_preds.append(parse(text))

sonnet_acc = sum(1 for l,p in zip(labels, sonnet_preds) if l==p) / len(labels)
print(f"Sonnet 4.6: accuracy={sonnet_acc:.2%}, avg_latency={np.mean(sonnet_lats):.0f}ms, errors={errors}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'Avg Latency':>12}")
print("-" * 60)
print(f"{'Qwen 2.5-3B + LoRA':<30} {lora_acc:>10.2%} {np.mean(lora_lats):>10.0f}ms")
print(f"{'Claude Sonnet 4.6':<30} {sonnet_acc:>10.2%} {np.mean(sonnet_lats):>10.0f}ms")
print("=" * 60)

# Save results
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "num_examples": len(examples),
    "lora_model": {"accuracy": lora_acc, "avg_latency_ms": float(np.mean(lora_lats)), "predictions": lora_preds},
    "sonnet_4_6": {"accuracy": sonnet_acc, "avg_latency_ms": float(np.mean(sonnet_lats)), "predictions": sonnet_preds, "errors": errors},
    "labels": labels,
}
with open("/Volumes/main/auto_research/autoresearch/quick_eval_v3_best.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to UC Volumes")

dbutils.notebook.exit(json.dumps({"lora_acc": lora_acc, "sonnet_acc": sonnet_acc}))
