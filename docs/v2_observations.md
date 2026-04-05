# Auto-Research v2 Observations

## Run Summary
- Agent: GPT 5.4 (databricks-gpt-5-4)
- Metric: Per-task weighted composite score
- Experiments: 10 (early stopped, patience=5)
- Runtime: 130 minutes
- Best composite: 0.708
- Best val_loss: 1.314

## THE AGENT ACTUALLY WORKED THIS TIME

GPT 5.4 returned valid JSON on every single call. Zero fallbacks. Every experiment was agent-proposed with a clear hypothesis. This is night and day from v1 where Llama 3.3 failed on every call.

## What GPT 5.4 tried (in order)

| # | Change | Composite | Sentiment | QA | Kept? |
|---|--------|-----------|-----------|-----|-------|
| 0 | Baseline | 0.596 | 24% | 92% | Yes |
| 1 | dropout 0.05->0.1 | 0.616 | 28% | 92% | **Yes** |
| 2 | rank 16->32 | 0.676 | 40% | 92% | **Yes** |
| 3 | alpha 32->128 | 0.708 | 44% | 96% | **Yes** (BEST) |
| 4 | rank 32->64 | 0.648 | 32% | 96% | No |
| 5 | rank 32->48 | 0.696 | 44% | 92% | No |
| 6 | lr 2e-4->1e-4 | 0.656 | 36% | 92% | No |
| 7 | rank 32->40 | 0.656 | 36% | 92% | No |
| 8 | alpha 128->192 | 0.668 | 36% | 96% | No |

## Key findings

### 1. GPT 5.4 had intelligent reasoning about multi-task trade-offs
Every proposal included reasoning like "sentiment classification is heavily weighted and underperforming, suggesting the current adapters may be too capacity-limited." The agent correctly identified that sentiment was the bottleneck task and focused its experiments there.

### 2. The agent found a sweet spot then plateaued
First 4 experiments: 3 consecutive improvements (dropout, rank, alpha). Each one specifically improved sentiment accuracy (24% -> 28% -> 40% -> 44%) while keeping QA stable (92-96%).

Experiments 4-8: All reverted. The agent kept trying variations around the sweet spot (rank 40, 48, 64; alpha 192; lower lr) but nothing beat the rank=32, alpha=128 config. Early stopping triggered correctly.

### 3. Sentiment accuracy nearly doubled: 24% -> 44%
This is the real story. v1 couldn't move sentiment accuracy at all (stuck at 58% with val_loss optimization). v2 with per-task composite scoring jumped sentiment from 24% to 44% in 4 experiments.

Wait, why is sentiment LOWER here (24-44%) vs v1 (58%)? Different eval methodology. v2 uses the model's raw generation output matched against expected labels. v1 used a simpler prompt format. The absolute numbers aren't directly comparable, but the TREND within v2 is clear.

### 4. Financial QA was already strong and stayed strong
QA accuracy was 92% at baseline and stayed 92-96% throughout. The agent never had to sacrifice QA performance to improve sentiment.

### 5. Data-aware agent > blind hyperparameter search
v1 (blind, val_loss only): 20 experiments, 6 kept, no sentiment improvement
v2 (data-aware, composite): 10 experiments, 4 kept, sentiment +20pp, early stopped
GPT 5.4 found better results in half the experiments because it could see what was underperforming.

## Loss curve data (per-experiment val_loss trajectory)
| Experiment | val_loss | Trend |
|---|---|---|
| 0 (baseline) | 1.387 | - |
| 1 (dropout) | 1.390 | +0.003 (slightly worse val_loss, but better composite) |
| 2 (rank 32) | 1.338 | -0.052 |
| 3 (alpha 128) | 1.314 | -0.024 (BEST) |
| 4 (rank 64) | 1.319 | +0.005 |
| 5 (rank 48) | 1.338 | +0.024 |
| 6 (lr 1e-4) | 1.355 | +0.041 |
| 7 (rank 40) | 1.333 | +0.019 |
| 8 (alpha 192) | 1.304 | -0.010 (lower val_loss but lower composite!) |

### Interesting: experiment 8 had the lowest val_loss (1.304) but wasn't kept
Because the composite score (0.668) was lower than the best (0.708). Lower val_loss doesn't mean better per-task performance. This validates the decision to use composite scoring in v2.
