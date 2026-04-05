"""
Download and preprocess financial datasets into instruction-tuning format
for LoRA fine-tuning Qwen 3.5 4B on Databricks serverless GPUs.

This script is READ-ONLY in the auto-research loop — the agent never modifies it.
Re-running produces the same output (idempotent).
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


SYSTEM_PROMPT = "You are a financial analyst assistant."

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TASK_SENTIMENT = "sentiment_classification"
TASK_QA = "financial_qa"
TASK_SUMMARIZATION = "summarization"
TASK_ENTITY_EXTRACTION = "entity_extraction"


# ---------------------------------------------------------------------------
# Dataset loaders — each returns a list of conversation dicts
# ---------------------------------------------------------------------------

def _msg(user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_type": "",
    }


def load_financial_phrasebank() -> list[dict]:
    """Sentiment-labeled financial sentences (Malo et al.)."""
    dataset_candidates = [
        ("SetFit/financial_phrasebank_allagree", None),
        ("zeroshot/twitter-financial-news-sentiment", None),
        ("financial_phrasebank", "sentences_allagree"),
    ]

    ds = None
    dataset_name = None
    for name, config in dataset_candidates:
        try:
            if config:
                ds = load_dataset(name, config, split="train")
            else:
                ds = load_dataset(name, split="train")
            dataset_name = name
            print(f"  Loaded {name}: {len(ds)} examples")
            break
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if ds is None:
        print("  WARNING: financial_phrasebank unavailable, skipping")
        return []

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    # Handle different column names across datasets
    text_col = "sentence" if "sentence" in ds.column_names else "text"
    label_col = "label" if "label" in ds.column_names else "label_text"

    examples = []
    for row in ds:
        sentence = str(row[text_col]).strip()
        raw_label = row[label_col]
        if isinstance(raw_label, int):
            label = label_map.get(raw_label, "neutral")
        else:
            label = str(raw_label).strip().lower()
            if label not in SENTIMENT_LABELS:
                label = "neutral"
        ex = _msg(
            f"Classify the sentiment of this financial text as positive, negative, or neutral:\n\n\"{sentence}\"",
            label,
        )
        ex["task_type"] = TASK_SENTIMENT
        examples.append(ex)

    return examples


def load_earnings_calls_qa() -> list[dict]:
    """Earnings call Q&A pairs."""
    dataset_candidates = [
        ("lamini/earnings-calls-qa", None),
        ("lamini/earnings_calls", None),
    ]

    ds = None
    for name, config in dataset_candidates:
        try:
            ds = load_dataset(name, config, split="train", )
            print(f"  Loaded {name}: {len(ds)} examples")
            break
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if ds is None:
        print("  WARNING: earnings calls QA dataset unavailable, generating synthetic fallback")
        return _synthetic_earnings_qa(2000)

    examples = []
    columns = ds.column_names

    if "question" in columns and "answer" in columns:
        for row in ds:
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            if not q or not a:
                continue
            ex = _msg(
                f"Based on this earnings call transcript, answer the following question:\n\nQuestion: {q}",
                a,
            )
            ex["task_type"] = TASK_QA
            examples.append(ex)
    elif "text" in columns:
        for row in ds:
            text = str(row["text"]).strip()
            if len(text) < 100:
                continue
            snippet = text[:1500]
            ex = _msg(
                f"Summarize this earnings call segment:\n\n{snippet}",
                _extractive_summary(text),
            )
            ex["task_type"] = TASK_SUMMARIZATION
            examples.append(ex)
    else:
        print(f"  Unexpected columns {columns}, generating synthetic fallback")
        return _synthetic_earnings_qa(2000)

    return examples


def load_sec_filings() -> list[dict]:
    """SEC 10-K filing excerpts for Q&A and entity extraction."""
    dataset_candidates = [
        ("JanosAudworx/sec-10k-filings", None),
        ("eloukas/edgar-corpus", None),
        ("nlpaueb/sec-filings", None),
    ]

    ds = None
    for name, config in dataset_candidates:
        try:
            ds = load_dataset(name, config, split="train", streaming=True)
            # Take a sample via streaming to avoid downloading full corpus
            rows = []
            for i, row in enumerate(ds):
                rows.append(row)
                if i >= 9999:
                    break
            ds = rows
            print(f"  Loaded {name}: {len(ds)} examples (streamed)")
            break
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if ds is None:
        print("  WARNING: SEC filings dataset unavailable, generating synthetic fallback")
        return _synthetic_sec_filings(3000)

    examples = []
    for row in ds:
        text = ""
        for key in ("text", "section_text", "content", "filing_text"):
            if key in row and row[key]:
                text = str(row[key]).strip()
                break

        if len(text) < 200:
            continue

        snippet = text[:2000]

        # Alternate between QA and entity extraction tasks
        if len(examples) % 2 == 0:
            ex = _msg(
                f"Based on this SEC filing excerpt, identify the key financial information:\n\n{snippet}",
                _extract_financial_info(text),
            )
            ex["task_type"] = TASK_ENTITY_EXTRACTION
        else:
            ex = _msg(
                f"Summarize the main points from this SEC filing excerpt:\n\n{snippet}",
                _extractive_summary(text),
            )
            ex["task_type"] = TASK_SUMMARIZATION

        examples.append(ex)

    return examples


def load_fingpt_sentiment() -> list[dict]:
    """FinGPT sentiment datasets — additional sentiment examples."""
    dataset_candidates = [
        ("FinGPT/fingpt-sentiment-train", None),
        ("zeroshot/twitter-financial-news-sentiment", None),
    ]

    ds = None
    for name, config in dataset_candidates:
        try:
            ds = load_dataset(name, config, split="train", )
            print(f"  Loaded {name}: {len(ds)} examples")
            break
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if ds is None:
        print("  WARNING: FinGPT sentiment dataset unavailable, skipping")
        return []

    label_map_str = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    label_map_int = {0: "negative", 1: "neutral", 2: "positive"}

    examples = []
    columns = ds.column_names

    text_col = next((c for c in ("input", "text", "sentence") if c in columns), None)
    label_col = next((c for c in ("output", "label", "sentiment") if c in columns), None)

    if not text_col or not label_col:
        print(f"  Unexpected columns {columns}, skipping")
        return []

    for row in ds:
        text = str(row[text_col]).strip()
        raw_label = row[label_col]

        if isinstance(raw_label, int):
            label = label_map_int.get(raw_label, "neutral")
        else:
            label = label_map_str.get(str(raw_label).strip().lower(), str(raw_label).strip().lower())

        if not text or len(text) < 10:
            continue

        ex = _msg(
            f"Classify the sentiment of this financial text as positive, negative, or neutral:\n\n\"{text}\"",
            label,
        )
        ex["task_type"] = TASK_SENTIMENT
        examples.append(ex)

    return examples


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------

def _deterministic_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


_SYNTHETIC_COMPANIES = [
    "Acme Corp", "TechVista Inc", "Global Dynamics", "Meridian Holdings",
    "Atlas Financial", "Pinnacle Systems", "Nexus Technologies", "Vanguard Industries",
    "Horizon Biotech", "Summit Energy", "Pacific Telecom", "Sterling Manufacturing",
]

_METRICS = [
    ("revenue", "$", "billion"), ("net income", "$", "million"),
    ("operating margin", "", "%"), ("earnings per share", "$", ""),
    ("free cash flow", "$", "million"), ("total debt", "$", "billion"),
    ("return on equity", "", "%"), ("gross profit", "$", "million"),
]


def _synthetic_earnings_qa(n: int) -> list[dict]:
    rng = random.Random(42)
    examples = []
    for i in range(n):
        company = rng.choice(_SYNTHETIC_COMPANIES)
        metric_name, prefix, suffix = rng.choice(_METRICS)
        value = round(rng.uniform(0.5, 50.0), 2)
        change = rng.choice(["increased", "decreased", "remained flat"])
        pct = round(rng.uniform(1, 30), 1)
        quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        year = rng.choice([2022, 2023, 2024])

        context = (
            f"During the {quarter} {year} earnings call, {company}'s CFO reported that "
            f"{metric_name} {change} by {pct}% year-over-year to {prefix}{value} {suffix}. "
            f"Management attributed this to {'strong demand' if change == 'increased' else 'challenging market conditions' if change == 'decreased' else 'stable operations'}."
        )
        ex = _msg(
            f"Based on this earnings call excerpt, what was {company}'s {metric_name}?\n\n{context}",
            f"{company}'s {metric_name} was {prefix}{value} {suffix}, which {change} by {pct}% year-over-year in {quarter} {year}.",
        )
        ex["task_type"] = TASK_QA
        examples.append(ex)
    print(f"  Generated {len(examples)} synthetic earnings QA examples")
    return examples


def _synthetic_sec_filings(n: int) -> list[dict]:
    rng = random.Random(43)
    examples = []
    for i in range(n):
        company = rng.choice(_SYNTHETIC_COMPANIES)
        year = rng.choice([2021, 2022, 2023])
        revenue = round(rng.uniform(1, 100), 1)
        net_income = round(revenue * rng.uniform(0.05, 0.25), 1)
        total_assets = round(revenue * rng.uniform(2, 5), 1)
        employees = rng.randint(5000, 200000)

        filing_text = (
            f"FORM 10-K — {company} — Fiscal Year Ended December 31, {year}\n\n"
            f"Item 1. Business: {company} is a {'technology' if 'Tech' in company else 'diversified'} company "
            f"with operations in multiple segments. The company employed approximately {employees:,} people.\n\n"
            f"Item 6. Selected Financial Data: Total revenues were ${revenue} billion. "
            f"Net income was ${net_income} billion. Total assets were ${total_assets} billion."
        )

        if i % 2 == 0:
            ex = _msg(
                f"Extract the key financial metrics from this SEC filing:\n\n{filing_text}",
                (
                    f"Company: {company}\n"
                    f"Fiscal Year: {year}\n"
                    f"Revenue: ${revenue} billion\n"
                    f"Net Income: ${net_income} billion\n"
                    f"Total Assets: ${total_assets} billion\n"
                    f"Employees: {employees:,}"
                ),
            )
            ex["task_type"] = TASK_ENTITY_EXTRACTION
        else:
            ex = _msg(
                f"Summarize the main points from this SEC filing excerpt:\n\n{filing_text}",
                (
                    f"{company} filed its 10-K for fiscal year {year}. "
                    f"The company generated ${revenue} billion in revenue and ${net_income} billion in net income, "
                    f"with total assets of ${total_assets} billion. It employed approximately {employees:,} people."
                ),
            )
            ex["task_type"] = TASK_SUMMARIZATION
        examples.append(ex)

    print(f"  Generated {len(examples)} synthetic SEC filing examples")
    return examples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = text.replace("\n", " ").split(". ")
    summary_sentences = [s.strip() for s in sentences[:max_sentences] if len(s.strip()) > 20]
    if not summary_sentences:
        return text[:500]
    return ". ".join(summary_sentences) + "."


def _extract_financial_info(text: str) -> str:
    lines = []
    text_lower = text.lower()
    for keyword in ["revenue", "income", "profit", "loss", "assets", "liabilities", "cash", "debt", "margin"]:
        if keyword in text_lower:
            idx = text_lower.index(keyword)
            start = max(0, idx - 20)
            end = min(len(text), idx + 100)
            snippet = text[start:end].strip().replace("\n", " ")
            lines.append(f"- {snippet}")
    if not lines:
        return f"Key information: {text[:300]}"
    return "Key financial metrics found:\n" + "\n".join(lines[:8])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_dataset(target_min: int = 20000, target_max: int = 50000) -> list[dict]:
    all_examples = []

    print("\n[1/4] Loading financial_phrasebank (sentiment)...")
    all_examples.extend(load_financial_phrasebank())

    print(f"\n[2/4] Loading earnings call transcripts (QA)...")
    all_examples.extend(load_earnings_calls_qa())

    print(f"\n[3/4] Loading SEC filings (entity extraction / summarization)...")
    all_examples.extend(load_sec_filings())

    print(f"\n[4/4] Loading FinGPT sentiment data...")
    all_examples.extend(load_fingpt_sentiment())

    print(f"\nTotal raw examples: {len(all_examples)}")

    # If we have too few, pad with more synthetics
    if len(all_examples) < target_min:
        deficit = target_min - len(all_examples)
        print(f"  Padding with {deficit} additional synthetic examples to reach {target_min}")
        extra_qa = _synthetic_earnings_qa(deficit // 2)
        extra_sec = _synthetic_sec_filings(deficit - deficit // 2)
        all_examples.extend(extra_qa)
        all_examples.extend(extra_sec)

    # If we have too many, downsample deterministically
    if len(all_examples) > target_max:
        print(f"  Downsampling from {len(all_examples)} to {target_max}")
        rng = random.Random(44)
        rng.shuffle(all_examples)
        all_examples = all_examples[:target_max]

    # Deterministic shuffle
    rng = random.Random(42)
    rng.shuffle(all_examples)

    return all_examples


def build_splits(examples: list[dict], val_ratio: float = 0.1) -> DatasetDict:
    split_idx = int(len(examples) * (1 - val_ratio))
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    def to_dataset(data: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "messages": [ex["messages"] for ex in data],
            "task_type": [ex["task_type"] for ex in data],
        })

    return DatasetDict({
        "train": to_dataset(train_data),
        "validation": to_dataset(val_data),
    })


def print_statistics(ds: DatasetDict) -> None:
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total examples:      {len(ds['train']) + len(ds['validation']):,}")
    print(f"  Train split:         {len(ds['train']):,}")
    print(f"  Validation split:    {len(ds['validation']):,}")

    all_types = list(ds["train"]["task_type"]) + list(ds["validation"]["task_type"])
    type_counts = {}
    for t in all_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n  Task type distribution:")
    for task_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_types)
        print(f"    {task_type:30s} {count:>7,}  ({pct:5.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare financial instruction-tuning dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/financial_instruct",
        help="Local directory to save the dataset (default: ./data/financial_instruct)",
    )
    parser.add_argument(
        "--uc-volume-path",
        type=str,
        default=None,
        help="Databricks UC Volumes path (e.g. /Volumes/catalog/schema/volume/data). If set, also copies output there.",
    )
    args = parser.parse_args()

    examples = assemble_dataset()
    ds = build_splits(examples)
    print_statistics(ds)

    # Save as HuggingFace Dataset (loadable with load_from_disk)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    print(f"\nSaved dataset to {output_path.resolve()}")

    # Also write a JSONL preview (first 5 examples) for quick inspection
    preview_path = output_path / "preview.jsonl"
    with open(preview_path, "w") as f:
        for ex in ds["train"].select(range(min(5, len(ds["train"])))):
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
    print(f"Wrote preview to {preview_path}")

    # Copy to UC Volumes if requested
    if args.uc_volume_path:
        uc_path = Path(args.uc_volume_path)
        try:
            uc_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(output_path), str(uc_path / "financial_instruct"), dirs_exist_ok=True)
            print(f"Copied dataset to UC Volume: {uc_path / 'financial_instruct'}")
        except Exception as e:
            print(f"WARNING: Could not copy to UC Volume path {uc_path}: {e}")
            print("  You can manually upload later with: databricks fs cp -r ...")


if __name__ == "__main__":
    main()
