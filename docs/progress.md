# Progress Log

## 2026-04-02: Project Build

### Completed
- [x] Project directory structure created
- [x] `pyproject.toml` with all dependencies
- [x] **Databricks Training Skill** (Phase 0)
  - `skill/SKILL.md` — skill definition with directives, hardware table, MLflow integration
  - `skill/references/hardware_guide.md` — GPU node types, VRAM, DBU costs
  - `skill/references/training_methods.md` — SFT + LoRA with trl/peft
  - `skill/references/mlflow_tracking.md` — MLflow experiment tracking
  - `skill/references/troubleshooting.md` — OOM, startup, tokenizer issues
  - `skill/scripts/train_sft_lora.py` — base LoRA SFT template
  - `skill/scripts/dataset_inspector.py` — pre-flight dataset validation
  - `skill/scripts/estimate_cost.py` — DBU cost estimator
- [x] **train.py** (Phase 1) — LoRA fine-tuning of Qwen 2.5-3B (Qwen 3.5 4B when available)
  - QLoRA (4-bit) on A10G GPU
  - Parameterized LoRA config + training hyperparams
  - Wall-clock timeout callback
  - MLflow logging, val_loss output line for orchestrator parsing
- [x] **prepare.py** (Phase 2) — financial data download + instruction-tuning format
  - FinancialPhraseBank (sentiment)
  - Earnings call Q&A
  - SEC filing summarization
  - Entity extraction (synthetic from financial text)
  - Outputs HuggingFace Dataset to disk
- [x] **Orchestrator** (Phase 3)
  - `orchestrator/agent_loop.py` — core auto-research loop
  - `orchestrator/submit_training.py` — Databricks Jobs API submission
  - `orchestrator/config.py` — all configuration in one place
- [x] **program.md** — agent instructions for the auto-research loop
- [x] **evaluate.py** — benchmark LoRA model vs FM API models (Llama 70B, DBRX)
- [x] **Databricks resources**
  - `setup_volumes.py` — UC Volumes creation
  - `cluster_spec.json` — GPU cluster spec (g5.xlarge)
  - `job_definition.json` — Databricks Job definition
- [x] **Databricks authentication** — both workspaces verified
  - e2-demo-field-eng (profile: e2-demo-west) — YES
  - e2-demo-west (profile: e2-demo-west-2) — YES
  - GPU confirmed: g5.xlarge (A10G, 24GB VRAM) available
- [x] `docs/decisions.md` — architectural decision log
- [x] Memory saved: writing style rules, SLM angle, HF Skills reference

### Completed (execution phase)
- [x] Run `prepare.py` — 50K examples (45K train, 5K val) from earnings calls, FinGPT, synthetic SEC
- [x] Upload data to UC Volumes on e2-demo-field-eng (main.auto_research) and free-workspace (workspace.auto_research)
- [x] Created test notebook (`02_test_training.py`) and uploaded to workspace
- [x] Fixed code review issues (dbfs: prefix, val_loss file output, BytesIO, enums)

### Blocked
- [ ] Test `train.py` on GPU cluster — e2-demo workspaces IP-blocked, free-workspace has no workers
- [ ] Need VPN connection or FEVM workspace to proceed with GPU testing

### Pending (after workspace access resolved)
- [ ] Test one full orchestrator loop iteration
- [ ] Run overnight (~100 experiments)
- [ ] Run `evaluate.py` — benchmark vs FM API models
- [ ] Write the article
- [ ] Create remaining notebooks

### Key file inventory
| File | Purpose | Lines |
|---|---|---|
| `train.py` | LoRA fine-tuning script (agent modifies this) | ~200 |
| `prepare.py` | Financial data download + preprocessing | ~400 |
| `evaluate.py` | Benchmark vs FM API models | ~250 |
| `program.md` | Agent instructions | ~100 |
| `orchestrator/agent_loop.py` | Core auto-research loop | ~250 |
| `orchestrator/submit_training.py` | Databricks job submission | ~130 |
| `orchestrator/config.py` | Configuration | ~50 |
| `skill/SKILL.md` | Training skill definition | ~100 |
