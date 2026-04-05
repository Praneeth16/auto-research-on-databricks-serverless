# Decision Log

All architectural decisions, actions, and steps taken during development.

## 2026-04-02: Project Kickoff

### Decision 1: Training Approach — LoRA Fine-Tuning (not training from scratch)
- **Context**: Karpathy's original auto-research trains a nano GPT from scratch. The GCP adaptation kept this approach.
- **Decision**: Fork auto-research to use LoRA fine-tuning of Qwen 3.5 4B instead.
- **Rationale**: (1) LoRA is what practitioners actually use. (2) A 4B model fine-tuned on domain data can compete with frontier models on narrow tasks. (3) More compelling article narrative: "small model beats big model."
- **Trade-off**: The auto-research agent now modifies LoRA config + hyperparams instead of model architecture. Narrower search space but more practical.

### Decision 2: Dataset — Blended Financial Corpus
- **Context**: Needed a domain-specific dataset large enough for meaningful training but focused enough to show domain specialization.
- **Decision**: Blend SEC EDGAR 10-K filings + earnings call transcripts + financial news. Instruction-tuning format.
- **Rationale**: Covers different registers of financial language (formal regulatory, conversational analyst Q&A, news). Instruction format enables direct evaluation.

### Decision 3: Evaluation — Head-to-Head vs Claude Opus/Sonnet 4.6
- **Context**: Need to prove the fine-tuned model is practically useful.
- **Decision**: Benchmark on FinancialPhraseBank (3-class sentiment) against Claude Opus 4.6 and Sonnet 4.6. Compare accuracy, latency, cost-per-inference.
- **Rationale**: The "small model beats frontier model on domain task" is the headline. Cost comparison makes the business case.

### Decision 4: Databricks Training Skill (adapted from HF Skills)
- **Context**: HuggingFace published a `hf-llm-trainer` skill that packages training knowledge for AI coding assistants.
- **Decision**: Create a Databricks-native equivalent that maps HF concepts (hf_jobs, Trackio, Hub) to Databricks (Jobs API, MLflow, UC Volumes).
- **Rationale**: Makes the training engine reusable beyond this project. Clean separation between the skill (how to train) and the auto-research loop (what to try).

### Decision 5: Writing Style
- **Context**: User provided 3 reference articles for writing style.
- **Decision**: No emdashes, no hyperboles, no AI slop. Emulate Raschka (hierarchical + precise), Chip (numbered frameworks + anecdotes), cyb3rops (intentional imperfection).
- **Reference articles**: cyb3rops, Raschka visual attention, Chip AI engineering pitfalls.

### Decision 6: Workspaces
- **Options**: e2-demo-field-eng.cloud.databricks.com or e2-demo-west.cloud.databricks.com
- **Decision**: Use whichever has serverless GPU availability. Check both.
- **Result**: Both authenticated. g5.xlarge (A10G, 24GB VRAM) confirmed available on e2-demo-field-eng.

### Decision 7: Workspace Selection
- **Attempted**: e2-demo-field-eng, e2-demo-west — blocked by IP ACL from current network
- **Attempted**: free-workspace — no worker environments (can't create clusters)
- **Resolution needed**: Connect to VPN, use FEVM workspace, or run notebooks from browser UI
- **Data uploaded to**: Both e2-demo-field-eng (main.auto_research) and free-workspace (workspace.auto_research) UC Volumes

### Decision 8: Frontier Model Comparison via FM API (not external APIs)
- **Context**: User wants to compare LoRA-tuned model against frontier models.
- **Decision**: Use Databricks Foundation Model API (Llama 3.1 70B, DBRX) instead of external Anthropic/OpenAI APIs.
- **Rationale**: Keeps the entire pipeline on-platform. Stronger Databricks narrative. No external API keys needed. FM API uses the same OpenAI-compatible interface.
