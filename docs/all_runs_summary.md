# All Auto-Research Runs Summary

## Run Comparison

| | v1 (Schedule) | v2 (GPT 5.4 Config) | v3 (GPT 5.4 Code Edit) |
|---|---|---|---|
| Agent | Llama 3.3 (failed, schedule fallback) | GPT 5.4 | GPT 5.4 |
| Search space | Hyperparameters only | Hyperparameters only | Full train.py editing |
| Optimization metric | val_loss | Composite (per-task weighted) | val_loss |
| Experiments | 20 (fixed) | 10 (early stop) | 24 (early stop) |
| Kept | 6 | 4 | 11 |
| Best val_loss | 1.280 | 1.314 | 1.254 |
| Improvement | -7.3% from 1.381 | -5.1% from 1.387 | -9.2% from 1.381 |
| Runtime | ~3.4 hours | ~2.2 hours | ~4 hours |
| Key discovery | lr + rank matter most | Agent reasons about multi-task trade-offs | MLP modules + budget optimization |

## v1 Best Config (Schedule Fallback)
- lr: 5e-4, rank: 64, alpha: 128, grad_accum: 8, weight_decay: 0.05

## v2 Best Config (GPT 5.4 Config Tuning)
- rank: 32, alpha: 128
- Sentiment accuracy: 24% -> 44% (composite: 0.596 -> 0.708)
- Agent correctly identified sentiment as bottleneck task

## v3 Best Config (GPT 5.4 Code Editing)
- target_modules: +gate_proj, up_proj, down_proj (MLP adapters)
- lr: 3.5e-4, dropout: 0.0, warmup: 0.0, weight_decay: 0.0
- eval_steps: 100, max_length: 768, batch_size: 8
- Peak VRAM: 18.19 GB (up from 10.74 GB baseline)

## v3 Agent Reasoning Highlights
- Exp 2: "increase adaptation capacity" by adding MLP modules -> biggest structural change
- Exp 5: "remove regularization hurting short budget" -> dropout 0.0
- Exp 8: "warmup wasting limited optimization window" -> warmup 0.0
- Exp 10: "reduce eval overhead, more training steps" -> eval_steps 100
- Exp 12: "reduce compute, fit more steps" -> max_length 768
- Exp 17: "improve gradient quality" -> batch_size 8

## Evaluation Results (all on same 50-example sentiment eval set)
- Qwen 2.5-3B baseline (no LoRA): ~47%
- v1 best (schedule fallback, lr=5e-4, rank=64): 58%
- v2 best (GPT 5.4, composite metric, rank=32, alpha=128): 44% (different eval method, composite 0.708)
- v3 best (GPT 5.4, code editing, MLP modules): **56%**, val_loss **1.256**
- Claude Sonnet 4.6 (FM API): **78%**

## Key Insight
val_loss improved 9.2% across iterations. Sentiment accuracy stayed at 56-58%.
The improvement went to overall language modeling, not the specific sentiment task (9% of training data).
To close the gap with Sonnet: need more sentiment data, bigger model, or sentiment-specific metric.
