# Autonomous LoRA Fine-Tuning on Databricks: Running Karpathy's Auto-Research on Financial Text

*54 experiments, 3 iterations, $30 in compute. The agent discovered that MLP adapters matter more than learning rates.*

## The experiment that changed my approach

I ran 20 LoRA fine-tuning experiments on financial text, fully automated, overnight on a Databricks GPU. An agent proposed changes to the training config, each experiment ran for 5 minutes, and the loop kept improvements and reverted failures. By morning, val_loss had dropped 7.3%.

Then I checked sentiment classification accuracy against Claude Sonnet 4.6. The fine-tuned model: 58%. Sonnet: 78%. Same as before the 20 experiments. val_loss improved. The actual task I cared about didn't move at all.

The training data was 90% financial Q&A, 9% sentiment, 1% entity extraction. The loop optimized val_loss across the blend. Most of that improvement went to Q&A. Sentiment was a rounding error.

That single result changed the entire project. I rebuilt the loop twice more, each time fixing what the previous run taught me. By the third iteration, the agent was editing the actual training source code, proposing architectural changes I wouldn't have tried manually, and reasoning about wall-clock efficiency as a training constraint. Here's how it played out.

## What auto-research is

Andrej Karpathy published [auto-research](https://github.com/karpathy/autoresearch) in early 2025. Give an LLM agent access to your training script. Let it propose one change at a time. Run each experiment for a fixed time budget. Keep improvements, revert failures. No human in the loop.

The one-change-at-a-time constraint is the key insight. Without it, you can't attribute improvements to specific decisions. The agent builds up a history of what worked and what didn't, and uses that to inform the next proposal.

I adapted auto-research for LoRA fine-tuning of Qwen 2.5-3B on financial text, running on Databricks serverless GPUs with GPT 5.4 as the agent. The base model uses 10.7 GB of VRAM with QLoRA (4-bit quantization), leaving headroom on an A10G (24 GB). Each experiment runs for 5 minutes. The training data is 50,000 instruction-tuning examples blended from earnings call transcripts (860K raw, downsampled), FinGPT financial news sentiment (77K), and synthetic SEC filing tasks (3K).

The dataset imbalance (90% Q&A, 9% sentiment, 1% extraction) turned out to be the most important detail of the whole project.

## Iteration 1: the hyperparameter sweep that missed the point

20 experiments, 5 minutes each, 3.4 hours total. The agent (Llama 3.3 70B via Foundation Model API) couldn't return valid JSON consistently, so the loop fell back to a predefined schedule that systematically tested learning rate, LoRA rank, batch size, and other knobs.

6 out of 20 changes improved val_loss:

| # | Change | val_loss | Kept? |
|---|--------|----------|-------|
| 0 | Baseline (lr=2e-4, rank=16) | 1.381 | Yes |
| 1 | **lr: 2e-4 -> 5e-4** | **1.315** | **Yes** |
| 4 | **rank: 16 -> 32** | **1.308** | **Yes** |
| 5 | **rank: 32 -> 64** | **1.295** | **Yes** |
| 10 | **grad_accum: 4 -> 8** | **1.280** | **Yes** |
| 14 | **weight_decay: 0.01 -> 0.05** | **1.280** | **Yes** |

Learning rate had the single biggest impact. Doubling it from 2e-4 to 5e-4 dropped val_loss by 0.066, more than any other change. Going lower (1e-4, 5e-5) made things dramatically worse. The default 2e-4 was too conservative for this dataset.

The 14 reverted experiments are informative too. Sequence length (512, 2048) didn't help. Scheduler type didn't matter. Optimizer variant didn't matter. Dropout didn't matter. Batch size 8 caused an OOM. For LoRA fine-tuning on this financial data, learning rate and rank were the only knobs worth turning.

**But val_loss improvement didn't translate to task accuracy.** 58% on sentiment before, 58% after. The loop optimized what it measured, and val_loss on a 90% Q&A dataset mostly measures Q&A performance. Sentiment was invisible to the metric.

## Iteration 2: teaching the agent about the data

I rebuilt the loop with three changes. Switched the agent to GPT 5.4 (reliable JSON output). Replaced val_loss with a per-task weighted composite score, weighting sentiment at 0.5 (the task I actually cared about). Fed the agent the data distribution and per-task accuracy after every experiment.

The agent's reasoning changed immediately. Instead of "try a higher learning rate," it started saying things like "sentiment classification is heavily weighted and underperforming, suggesting the current adapters may be too capacity-limited to capture task-specific sentiment patterns under the imbalanced training mix."

10 experiments (early stopped after 5 consecutive failures), 4 kept:

| # | GPT 5.4's proposal | Composite | Sentiment acc |
|---|---|---|---|
| 0 | Baseline | 0.596 | 24% |
| 1 | Increase LoRA dropout for regularization | 0.616 | 28% |
| 2 | **Increase rank for adapter capacity** | **0.676** | **40%** |
| 3 | **Increase alpha to strengthen adapters** | **0.708** | **44%** |

Sentiment accuracy nearly doubled in 4 experiments. The agent correctly identified adapter capacity as the bottleneck for the underrepresented task and fixed it with two targeted changes.

One finding I didn't expect: experiment 8 had the lowest val_loss of any run (1.304) but wasn't kept, because its composite score (0.668) was lower than the best (0.708). Lower val_loss and higher per-task accuracy can diverge. That's exactly why I needed to switch metrics.

## Iteration 3: letting the agent edit the actual code

The first two iterations restricted the agent to tweaking config values. Karpathy's original auto-research does something more ambitious: the agent sees the full `train.py` source and can modify anything. Architecture, optimizer logic, training loop, all of it.

I aligned the loop with that design. GPT 5.4 receives the complete training script, proposes a modified version, and the system runs it via `exec()`. I also added three improvements from studying Karpathy's original code: fast-fail on NaN or loss > 100, exclude the first 10 warmup steps from the time budget, and a do-not-repeat ledger so the agent doesn't retry failed experiments.

24 experiments (early stopped), 11 kept. val_loss: 1.381 to 1.254, a 9.2% improvement.

The interesting part is what the agent tried:

| # | What GPT 5.4 changed in the code | val_loss | Agent's reasoning |
|---|---|---|---|
| 0 | Baseline | 1.381 | |
| 2 | **Added MLP adapter modules** (gate, up, down_proj) | **1.316** | "increase adaptation capacity" |
| 4 | lr: 2e-4 -> 3e-4 | 1.298 | "speed adaptation within 300s budget" |
| 5 | dropout: 0.05 -> 0.0 | 1.297 | "remove regularization hurting short budget" |
| 6 | lr: 3e-4 -> 4e-4 | 1.290 | "slightly faster adaptation" |
| 8 | **warmup: 0.03 -> 0.0** | **1.271** | "warmup wasting limited optimization window" |
| 10 | **eval_steps: 50 -> 100** | **1.259** | "reduce eval overhead, more training steps" |
| 12 | max_length: 1024 -> 768 | 1.259 | "reduce compute, fit more steps" |
| 15 | lr: 4e-4 -> 3.5e-4 | 1.257 | "slightly lower for generalization" |
| 16 | weight_decay: 0.01 -> 0.0 | 1.257 | "less regularization, faster fitting" |
| 17 | batch_size: 4 -> 8 | 1.254 | "improve gradient quality" |

Two things stand out.

**Adding MLP adapter modules** (experiment 2) was the second-biggest single improvement. This is a structural change to what LoRA adapts, not a number you turn. Iterations 1 and 2 couldn't try this because they only tuned config values. The agent had to see the code to propose it.

The agent also figured out that **with a 5-minute training budget, every second matters.** Removing warmup (experiment 8), reducing eval frequency (experiment 10), and shortening sequences (experiment 12) are all systems-level optimizations. The agent reasoned about wall-clock efficiency, not just model quality. A hyperparameter grid search wouldn't find these.

![val_loss trajectory across 3 iterations](../assets/val_loss_trajectory.png)
*Filled dots = kept experiments. v3 (green) found steady improvements through code-level changes that v1 (blue) couldn't reach with config tuning alone.*

## The full picture

| | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---|---|---|
| Agent | Llama 3.3 (failed) | GPT 5.4 | GPT 5.4 |
| Search space | Config values | Config values | **Full source code** |
| Metric | val_loss | Per-task composite | val_loss |
| Experiments | 20 | 10 | 24 |
| Best val_loss | 1.280 | 1.314 | **1.254** |
| Key discovery | lr + rank dominate | Data-aware agent > blind search | MLP adapters + budget optimization |

54 experiments total, ~10 hours of GPU time, roughly $30 in compute.

## Results: 3B model vs Sonnet 4.6

| Model | Sentiment Accuracy | val_loss |
|-------|----------|----------|
| Qwen 2.5-3B baseline (no fine-tuning) | ~47% | |
| Qwen 2.5-3B + LoRA (single 2-min run) | 58% | 1.386 |
| Qwen 2.5-3B + LoRA (iteration 1, 20 experiments) | 58% | 1.280 |
| Qwen 2.5-3B + LoRA (iteration 3, code editing, MLP adapters) | 56% | 1.256 |
| Claude Sonnet 4.6 (FM API) | 78% | |

The sentiment accuracy tells a consistent story: the 3B model lands around 56-58%, Sonnet stays at 78%. The 22-point gap didn't close.

That's not because auto-research failed. val_loss dropped 9.2%. The model genuinely improved at financial language modeling. But sentiment classification needs more than general improvement on a blended dataset where sentiment is 9% of examples. The agent discovered this in iteration 2 when it switched to per-task scoring and doubled sentiment accuracy within its own framework.

The honest conclusion: a 3B model with LoRA can't match Sonnet 4.6 on financial sentiment when sentiment is a minority task in the training data. You'd need more sentiment examples, a bigger model, or both. But auto-research is how you find that out in 10 hours of unattended GPU time instead of a week of manual work.

## What I learned

**What you measure is what you improve.** Iteration 1 optimized val_loss on a blended dataset. val_loss improved 7.3%. Sentiment accuracy stayed flat. Iteration 2 switched to a per-task composite score and sentiment accuracy nearly doubled. Same code, same data, different metric. If you're fine-tuning on multi-task data, evaluate per-task.

**A smarter agent finds better results in fewer experiments.** Llama 3.3 70B couldn't return valid JSON and fell back to a random schedule. GPT 5.4 proposed 4 targeted changes and found a better config than 20 random experiments. Agent quality matters more than experiment count.

**Code access unlocks changes that hyperparameter sweeps can't find.** The MLP adapter modules discovery (iteration 3, experiment 2) was the kind of architectural insight that no amount of learning rate tuning would produce. The agent needed to see and edit the LoRA config code to propose it. Karpathy's design, where the agent modifies the full training script, is genuinely more powerful than a config-only search space.

**With a fixed time budget, training efficiency is a hyperparameter.** The agent's decision to remove warmup, reduce eval frequency, and shorten sequences were all about squeezing more training into 5 minutes. Under a wall-clock constraint, these matter. The agent figured this out on its own.

## Making it work on Databricks

A few implementation details for anyone who wants to reproduce this.

**The infrastructure.** Databricks GPU ML Runtimes (15.4.x-gpu-ml) ship with CUDA, PyTorch, and transformers pre-installed. `torch.cuda.is_available()` works on the first try. The only packages to add are `peft`, `trl`, and `bitsandbytes`, via a cluster init script. Serverless GPU clusters start in 3 minutes and auto-terminate when idle. The agent LLM (GPT 5.4) runs on the same platform via Foundation Model API. One workspace, one notebook, one GPU.

```
┌─────────────────────────────────────────────────────────┐
│  Databricks Workspace                                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Serverless GPU Cluster (g5.xlarge / A10G)      │    │
│  │                                                  │    │
│  │  Notebook: Auto-Research Loop                    │    │
│  │    1. Call agent ─► Foundation Model API (GPT 5.4)│    │
│  │    2. Agent returns modified train.py             │    │
│  │    3. Run training (5 min)                       │    │
│  │    4. Measure val_loss                           │    │
│  │    5. Keep or revert                             │    │
│  │    6. Repeat                                     │    │
│  └──────────────┬──────────────────────────────────┘    │
│                  │                                       │
│  ┌───────────────▼───────────────────────────────┐      │
│  │  Unity Catalog Volumes                        │      │
│  │  - Training data (50K instruction examples)   │      │
│  │  - LoRA adapters (~30MB each)                 │      │
│  │  - Results TSV + experiment history           │      │
│  └───────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

**Things that broke.** In roughly the order we hit them:

1. `%pip install peft trl bitsandbytes` upgrades transformers to a version that breaks PyTorch detection after `restartPython()`. Fix: use a cluster init script instead.

2. `BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16)` fails in some runtime contexts. Pass `"bfloat16"` as a string.

3. `SFTTrainer` renamed `tokenizer` to `processing_class` in recent trl versions.

4. UC Volumes don't support `open(path, "a")`. Write to `/tmp/` and copy.

5. The Databricks runtime auto-enables MLflow tracking. Subprocesses inherit this but lack the auth context, causing every training run to crash. Fix: run training via `exec()` in the notebook process, or set `MLFLOW_TRACKING_URI=file:///tmp/mlflow_noop` in the subprocess env.

6. SparkPythonTask can't read scripts from UC Volumes paths. Use DBFS or notebook tasks.

## Try it yourself

The full code is on GitHub: [Praneeth16/auto-research-on-databricks-serverless](https://github.com/Praneeth16/auto-research-on-databricks-serverless)

To run it on your own Databricks workspace:

1. Create a GPU cluster (g5.xlarge, DBR 15.4 ML) with an init script that installs `peft trl bitsandbytes accelerate datasets`
2. Upload your dataset to UC Volumes using `prepare.py`
3. Start with `notebooks/03_auto_research_v1.py` (config tuning) to establish a baseline
4. Graduate to `notebooks/05_auto_research_v3.py` (full code editing) for deeper optimization

To adapt for your own domain, replace the financial datasets in `prepare.py` with your data. Keep the instruction-tuning format. If your dataset has a task type column, the v2 loop auto-detects it and evaluates per-task. The LoRA adapter from our best experiment is ~30 MB, loadable on any machine with 18 GB of VRAM.

## What I'd do next

**Rebalance the training mix.** Sentiment was 9% of training data and never got enough signal. I'd either upsample sentiment examples or add a data mixing ratio as a tunable parameter the agent can adjust.

**Use a longer time budget.** Five minutes is tight. At 10 minutes per experiment, the model processes roughly twice the data, and the warmup overhead becomes proportionally smaller.

**Try a bigger base model.** Qwen 2.5-7B would need more VRAM but might close the gap with Sonnet on sentiment.

**Run the loop as a scheduled job.** Right now it's a notebook. With Databricks Jobs, you could schedule it to run every night, building on the previous best adapter each time.
