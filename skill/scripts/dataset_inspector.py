# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=3.0.0",
# ]
# ///
"""
Pre-flight dataset validation for LoRA SFT training.

Checks dataset format, completeness, and statistics before GPU resources
are allocated. Outputs a verdict: READY, NEEDS_MAPPING, or INCOMPATIBLE.

Usage:
    python dataset_inspector.py /Volumes/catalog/schema/autoresearch/data/train.jsonl
    python dataset_inspector.py /path/to/data.jsonl --format conversational
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


CONVERSATIONAL_REQUIRED = {"messages"}
TEXT_REQUIRED = {"text"}
VALID_ROLES = {"system", "user", "assistant"}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: Line {i} is not valid JSON: {e}")
                return []
    return records


def detect_format(records: list[dict]) -> str | None:
    if not records:
        return None
    sample = records[0]
    if "messages" in sample and isinstance(sample["messages"], list):
        return "conversational"
    if "text" in sample and isinstance(sample["text"], str):
        return "text"
    return None


def validate_conversational(records: list[dict]) -> list[str]:
    issues = []
    role_counts = Counter()
    empty_content = 0
    bad_roles = Counter()

    for i, record in enumerate(records):
        if "messages" not in record:
            issues.append(f"Row {i}: missing 'messages' field")
            continue
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            issues.append(f"Row {i}: 'messages' is empty or not a list")
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                issues.append(f"Row {i}: message is not a dict")
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_counts[role] += 1
            if role not in VALID_ROLES:
                bad_roles[role] += 1
            if not content or not content.strip():
                empty_content += 1

    if bad_roles:
        for role, count in bad_roles.items():
            issues.append(f"Invalid role '{role}' found {count} times")
    if empty_content > 0:
        issues.append(f"{empty_content} messages have empty content")

    return issues


def validate_text(records: list[dict]) -> list[str]:
    issues = []
    empty_count = 0

    for i, record in enumerate(records):
        if "text" not in record:
            issues.append(f"Row {i}: missing 'text' field")
            continue
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            empty_count += 1

    if empty_count > 0:
        issues.append(f"{empty_count} rows have empty text")

    return issues


def compute_stats(records: list[dict], fmt: str) -> dict:
    lengths = []
    for record in records:
        if fmt == "conversational":
            total = sum(len(m.get("content", "")) for m in record.get("messages", []))
            lengths.append(total)
        elif fmt == "text":
            lengths.append(len(record.get("text", "")))

    if not lengths:
        return {"count": 0}

    lengths.sort()
    return {
        "count": len(lengths),
        "avg_chars": round(sum(lengths) / len(lengths)),
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "median_chars": lengths[len(lengths) // 2],
        "p95_chars": lengths[int(len(lengths) * 0.95)],
    }


def inspect(path: str, expected_format: str | None = None) -> dict:
    result = {
        "path": path,
        "verdict": "INCOMPATIBLE",
        "issues": [],
        "stats": {},
        "detected_format": None,
    }

    if not Path(path).exists():
        result["issues"].append(f"File not found: {path}")
        return result

    records = load_jsonl(path)
    if not records:
        result["issues"].append("No valid records found")
        return result

    detected = detect_format(records)
    result["detected_format"] = detected

    if detected is None:
        result["issues"].append(
            f"Could not detect format. First record keys: {list(records[0].keys())}"
        )
        columns = set(records[0].keys())
        if columns & CONVERSATIONAL_REQUIRED or columns & TEXT_REQUIRED:
            result["verdict"] = "NEEDS_MAPPING"
        return result

    if expected_format and detected != expected_format:
        result["issues"].append(
            f"Expected '{expected_format}' format but detected '{detected}'"
        )
        result["verdict"] = "NEEDS_MAPPING"
        return result

    if detected == "conversational":
        issues = validate_conversational(records)
    else:
        issues = validate_text(records)

    result["issues"] = issues
    result["stats"] = compute_stats(records, detected)

    if not issues:
        result["verdict"] = "READY"
    elif all("empty" in issue.lower() for issue in issues):
        result["verdict"] = "NEEDS_MAPPING"
    else:
        result["verdict"] = "INCOMPATIBLE"

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for LoRA SFT training")
    parser.add_argument("path", help="Path to JSONL dataset file")
    parser.add_argument(
        "--format",
        choices=["conversational", "text"],
        default=None,
        help="Expected dataset format",
    )
    args = parser.parse_args()

    result = inspect(args.path, args.format)

    print(f"\n{'='*60}")
    print(f"Dataset Inspection: {result['path']}")
    print(f"{'='*60}")
    print(f"Detected format: {result['detected_format']}")
    print(f"Verdict: {result['verdict']}")

    if result["stats"]:
        print(f"\nStatistics:")
        for k, v in result["stats"].items():
            print(f"  {k}: {v}")

    if result["issues"]:
        print(f"\nIssues ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo issues found.")

    print(f"\n{result['verdict']}")

    if result["verdict"] == "INCOMPATIBLE":
        sys.exit(1)
    elif result["verdict"] == "NEEDS_MAPPING":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
