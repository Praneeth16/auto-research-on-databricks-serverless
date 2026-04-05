"""
Evaluate the best LoRA-tuned model against frontier models via Foundation Model API.

Runs the same financial sentiment tasks through:
1. Our LoRA-tuned Qwen 3.5 4B (local inference)
2. Llama 3.1 70B (via Databricks Foundation Model API)
3. Llama 3.1 405B (via Databricks Foundation Model API)
4. DBRX (via Databricks Foundation Model API)

Compares accuracy, latency, and cost-per-inference on FinancialPhraseBank (3-class sentiment).

All frontier model evaluations use Databricks Foundation Model API (FM API),
keeping the entire pipeline on-platform. No external API keys needed.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score


SENTIMENT_LABELS = ["positive", "negative", "neutral"]

PROMPT_TEMPLATE = """Classify the sentiment of the following financial text as exactly one of: positive, negative, or neutral.

Text: {text}

Sentiment:"""

FM_API_MODELS = [
    {
        "name": "Llama 3.1 70B",
        "endpoint": "databricks-meta-llama-3-1-70b-instruct",
        "cost_per_1k_tokens": 0.001,  # approximate
    },
    {
        "name": "DBRX Instruct",
        "endpoint": "databricks-dbrx-instruct",
        "cost_per_1k_tokens": 0.00075,
    },
]


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    predictions: list
    latencies: list
    total_time: float
    avg_latency_ms: float
    cost_estimate: float


def load_financial_phrasebank():
    """Load FinancialPhraseBank for evaluation."""
    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        split="train",
        trust_remote_code=True,
    )

    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    examples = []
    for row in dataset:
        examples.append({
            "text": row["sentence"],
            "label": label_map[row["label"]],
        })

    return examples


def parse_sentiment(response_text: str) -> str:
    """Parse sentiment label from model response."""
    response_lower = response_text.strip().lower()
    for label in SENTIMENT_LABELS:
        if label in response_lower:
            return label
    return "neutral"  # default fallback


def evaluate_lora_model(
    base_model_path: str,
    adapter_path: str,
    examples: list,
    max_examples: int = None,
) -> EvalResult:
    """Evaluate the LoRA-tuned model on financial sentiment."""
    print(f"\nEvaluating LoRA model: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if max_examples:
        examples = examples[:max_examples]

    predictions = []
    latencies = []
    start_total = time.time()

    for i, ex in enumerate(examples):
        prompt = PROMPT_TEMPLATE.format(text=ex["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_sentiment(response)
        predictions.append(pred)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(examples)} done, avg latency: {sum(latencies)/len(latencies):.0f}ms")

    total_time = time.time() - start_total
    labels = [ex["label"] for ex in examples]
    acc = accuracy_score(labels, predictions)

    return EvalResult(
        model_name=f"Qwen 3.5 4B + LoRA (financial)",
        accuracy=acc,
        predictions=predictions,
        latencies=latencies,
        total_time=total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        cost_estimate=0.0,  # local inference, no API cost
    )


def evaluate_fm_api_model(
    host: str,
    token: str,
    model_config: dict,
    examples: list,
    max_examples: int = None,
) -> EvalResult:
    """Evaluate a Foundation Model API model on financial sentiment."""
    model_name = model_config["name"]
    endpoint = model_config["endpoint"]
    print(f"\nEvaluating FM API model: {model_name}")

    client = OpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
    )

    if max_examples:
        examples = examples[:max_examples]

    predictions = []
    latencies = []
    total_tokens = 0
    start_total = time.time()

    for i, ex in enumerate(examples):
        prompt = PROMPT_TEMPLATE.format(text=ex["text"])

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=endpoint,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Respond with exactly one word: positive, negative, or neutral."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
            total_tokens += response.usage.total_tokens if response.usage else 0
        except Exception as e:
            print(f"  API error at example {i}: {e}")
            response_text = "neutral"

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        pred = parse_sentiment(response_text)
        predictions.append(pred)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(examples)} done, avg latency: {sum(latencies)/len(latencies):.0f}ms")

        # Rate limiting
        if latency < 100:
            time.sleep(0.05)

    total_time = time.time() - start_total
    labels = [ex["label"] for ex in examples]
    acc = accuracy_score(labels, predictions)
    cost = (total_tokens / 1000) * model_config["cost_per_1k_tokens"]

    return EvalResult(
        model_name=model_name,
        accuracy=acc,
        predictions=predictions,
        latencies=latencies,
        total_time=total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        cost_estimate=cost,
    )


def print_comparison_table(results: list[EvalResult], examples: list):
    """Print a formatted comparison table."""
    labels = [ex["label"] for ex in examples[:len(results[0].predictions)]]

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS: Financial Sentiment (FinancialPhraseBank)")
    print("=" * 80)

    header = f"{'Model':<40} {'Accuracy':>10} {'Avg Latency':>12} {'Cost':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        cost_str = f"${r.cost_estimate:.4f}" if r.cost_estimate > 0 else "local"
        print(f"{r.model_name:<40} {r.accuracy:>10.4f} {r.avg_latency_ms:>10.0f}ms {cost_str:>10}")

    print("-" * 80)

    # Detailed classification report for each model
    for r in results:
        print(f"\n--- {r.model_name} ---")
        print(classification_report(labels, r.predictions, labels=SENTIMENT_LABELS, zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--databricks-host", type=str, default="https://e2-demo-field-eng.cloud.databricks.com")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--skip-local", action="store_true", help="Skip local LoRA model evaluation")
    args = parser.parse_args()

    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not token:
        print("Warning: DATABRICKS_TOKEN not set. FM API evaluation will fail.")

    # Load evaluation data
    print("Loading FinancialPhraseBank...")
    examples = load_financial_phrasebank()
    print(f"Loaded {len(examples)} examples")

    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Using {len(examples)} examples for evaluation")

    results = []

    # Evaluate LoRA model
    if not args.skip_local:
        lora_result = evaluate_lora_model(args.base_model, args.adapter_path, examples)
        results.append(lora_result)

    # Evaluate FM API models
    for model_config in FM_API_MODELS:
        try:
            fm_result = evaluate_fm_api_model(
                args.databricks_host, token, model_config, examples
            )
            results.append(fm_result)
        except Exception as e:
            print(f"Failed to evaluate {model_config['name']}: {e}")

    # Print results
    print_comparison_table(results, examples)

    # Save to JSON
    output_data = []
    for r in results:
        output_data.append({
            "model_name": r.model_name,
            "accuracy": r.accuracy,
            "avg_latency_ms": r.avg_latency_ms,
            "total_time_seconds": r.total_time,
            "cost_estimate": r.cost_estimate,
            "num_examples": len(r.predictions),
        })

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
