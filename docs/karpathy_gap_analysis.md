# Gap Analysis: Our Implementation vs Karpathy's Original

## Three major design deviations we found

### 1. We restricted the agent to hyperparameter tuning (Karpathy lets it edit anything)

Karpathy's agent sees the full `train.py` source and can modify model architecture, optimizer implementation, training loop logic, data preprocessing, anything. The only constraint is: don't touch `prepare.py` (evaluation + data loading) and keep within VRAM budget.

We gave our agent a menu of config values to pick from: `{"param": "learning_rate", "value": 5e-4}`. This turns auto-research into fancy hyperparameter search. The agent can't try a different attention pattern, add gradient clipping, implement a custom learning rate warmup, or restructure the training loop.

**Fix in v3**: Agent receives and returns the full train.py source code.

### 2. No fast-fail on numerical instability

Karpathy's train.py exits immediately when loss > 100 or is NaN (line 570-572). Our experiments burned the full 5-minute budget even on diverging runs.

This matters because: at lr=5e-5 (experiment 3 in v1), val_loss hit 2.29. The model was clearly diverging after a few steps, but we ran the full 5 minutes before discovering that.

**Fix in v3**: Add on_log callback that checks loss every logging step. Exit early on NaN or loss > 100.

### 3. Warmup steps eat into the training budget

Karpathy excludes the first 10 steps from the time budget because they include torch.compile and CUDA graph warmup. Our TimeoutCallback starts counting from step 0.

On a cold model load with QLoRA, the first few steps include:
- bitsandbytes quantization initialization
- CUDA kernel compilation
- Graph optimization

This overhead is ~30-60 seconds on A10G. With a 5-minute budget, that's 10-20% of training time wasted on non-training work.

**Fix in v3**: TimeoutCallback skips first 10 steps before counting.

## Three medium gaps

### 4. No do-not-repeat ledger
multiautoresearch maintains `research/do-not-repeat.md` so the planner avoids re-running failed experiments. Our agent has no memory of what specific code changes failed (only config values from history).

**Fix in v3**: Maintain /tmp/do_not_repeat.md with failed experiment descriptions.

### 5. Temperature too high
multiautoresearch uses temperature 0.1 for the experiment-worker agent. We used 0.4-0.7. Lower temperature means more deterministic code edits, fewer hallucinated changes.

**Fix in v3**: Temperature 0.1.

### 6. Agent doesn't see the actual training code
In our v1/v2, the agent only sees config values. It doesn't know what the training loop looks like, what optimizer is actually being used, or what the data loading code does. Karpathy's agent reads the full train.py before every experiment.

**Fix in v3**: Agent sees full train.py source in every prompt.

## What we got right

- One change per experiment (matches Karpathy exactly)
- Results in TSV format with tabs (matches Karpathy's advice about commas breaking)
- Keep on strict improvement, revert on equal or worse
- Fixed time budget per experiment
- Read-only evaluation (val_loss computed by the trainer, agent can't game it)

## Version evolution story (for the article)

| Version | Agent | Metric | Search Space | Key Finding |
|---------|-------|--------|-------------|-------------|
| v1 | Llama 3.3 70B | val_loss | Config values only | Agent JSON calls failed, fell back to random. val_loss -7.3% but sentiment acc flat. |
| v2 | GPT 5.4 | Composite (per-task) | Config values only | Multi-task aware, dynamic budget, adaptive task detection |
| v3 | GPT 5.4 | val_loss + per-task | Full train.py editing | Aligned with Karpathy: fast-fail, warmup exclusion, do-not-repeat, code-level changes |
