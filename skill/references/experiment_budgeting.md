# Dynamic Experiment Budgeting

Best practices for determining how many experiments to run in an auto-research loop, based on task type, data characteristics, and convergence behavior.

## Rules of Thumb

### Minimum experiments by task complexity
| Task Type | Min Experiments | Rationale |
|-----------|----------------|-----------|
| Single-task (e.g., sentiment only) | 10-15 | Fewer knobs matter; lr + rank usually sufficient |
| Multi-task (blended dataset) | 20-30 | Need to explore per-task trade-offs |
| Multi-task + data rebalancing | 30-50 | Agent may need to adjust data mixing ratios |

### Convergence detection
Stop early if:
- No improvement in the last `patience` consecutive experiments (default: 5)
- Best metric hasn't improved by more than `min_delta` (default: 0.5%) in the last `lookback` experiments (default: 8)
- VRAM OOM occurs on 3+ consecutive experiments (search space exhausted for this GPU)

### Budget estimation formula
```
estimated_experiments = base_experiments × task_multiplier × data_size_factor

where:
  base_experiments = 15
  task_multiplier = number of distinct task types in dataset (1-5)
  data_size_factor = 1.0 if <50K examples, 1.5 if 50K-200K, 2.0 if >200K
```

For our financial dataset (3 task types, 50K examples):
  15 × 3 × 1.0 = 45 experiments (with early stopping via patience=5)

### Per-task evaluation
When training on multi-task data, evaluate each task separately after every experiment:
1. Sample N examples per task from validation set (N=20-50)
2. Run inference, compute task-specific accuracy
3. Report weighted score: `sum(task_weight × task_accuracy)`
4. Feed per-task breakdown to the agent so it can reason about trade-offs

### Data-aware agent
The agent should receive:
- Task distribution in training data (% per task type)
- Per-task accuracy after each experiment
- Sample examples from underperforming tasks
- VRAM headroom for the current config
