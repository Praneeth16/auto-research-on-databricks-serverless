# Auto-Research on Databricks Serverless GPUs

Autonomous LoRA fine-tuning of Qwen 2.5-3B on financial text, using Karpathy's auto-research approach on Databricks serverless GPUs.

An AI agent (GPT 5.4) ran 54 experiments across 3 iterations, cutting val_loss by 9.2% and discovering that adding MLP adapter modules matters more than tuning learning rates.

Read the full article: [docs/article.md](docs/article.md)

## Quick Start

### Prerequisites
- Databricks workspace with GPU cluster access (g5.xlarge / A10G recommended)
- Cluster init script to install: `peft trl bitsandbytes accelerate datasets`
- Databricks CLI authenticated

### 1. Prepare data
```bash
python prepare.py --output-dir ./data/financial_instruct
```

### 2. Upload to Databricks
```bash
python databricks_resources/setup_volumes.py --profile your-profile
databricks fs cp -r ./data/financial_instruct dbfs:/Volumes/main/auto_research/autoresearch/data/financial_instruct --profile your-profile --overwrite
```

### 3. Run auto-research
Upload and run one of the notebooks on a GPU cluster:
- `notebooks/03_auto_research_v1.py` - Hyperparameter tuning (config values)
- `notebooks/04_auto_research_v2.py` - Multi-task aware, per-task composite scoring
- `notebooks/05_auto_research_v3.py` - Full train.py code editing (Karpathy-aligned)

## Project Structure

```
├── train.py                     # Base training script (agent modifies this in v3)
├── prepare.py                   # Dataset download and preprocessing
├── evaluate.py                  # Benchmark against FM API models
├── program.md                   # Agent instructions
├── pyproject.toml               # Dependencies
├── skill/                       # Databricks training skill
│   ├── SKILL.md
│   ├── references/              # Hardware, training methods, MLflow, troubleshooting guides
│   └── scripts/                 # Training templates, dataset inspector, cost estimator
├── orchestrator/                # Orchestration code (for programmatic job submission)
│   ├── agent_loop.py
│   ├── submit_training.py
│   └── config.py
├── notebooks/                   # Databricks notebooks (run these)
│   ├── 01_test_training.py      # Verify GPU + LoRA works
│   ├── 02_quick_eval.py         # Quick eval against Sonnet 4.6
│   ├── 03_auto_research_v1.py   # v1: config tuning
│   ├── 04_auto_research_v2.py   # v2: multi-task, GPT 5.4
│   └── 05_auto_research_v3.py   # v3: full code editing
├── databricks_resources/        # Cluster specs, job definitions, setup scripts
├── docs/                        # Article, results, observations
│   ├── article.md               # The full article
│   ├── all_runs_summary.md      # Results from all 54 experiments
│   └── *.tsv / *.json           # Raw experiment data
└── assets/                      # Charts and diagrams
```

## Results Summary

| Iteration | Agent | Experiments | Best val_loss | Key Discovery |
|-----------|-------|-------------|---------------|---------------|
| v1 | Llama 3.3 (fallback schedule) | 20 | 1.280 | lr + rank dominate |
| v2 | GPT 5.4 (config tuning) | 10 | 1.314 | Data-aware agent + per-task scoring |
| v3 | GPT 5.4 (code editing) | 24 | 1.254 | MLP adapters + budget optimization |

## Adapting for Your Domain

1. Replace the financial datasets in `prepare.py` with your data
2. Keep the instruction-tuning format (`messages` with system/user/assistant)
3. If your dataset has a task type column, v2/v3 auto-detect it
4. Adjust `TRAINING_BUDGET_SECONDS` and cluster type for your compute budget

## References

- [Karpathy's auto-research](https://github.com/karpathy/autoresearch)
- [HuggingFace auto-research variant](https://github.com/mishig25/hf-autoresearch)
- [Multi-agent auto-research](https://github.com/burtenshaw/multiautoresearch)
- [HuggingFace training skills](https://github.com/huggingface/skills)
