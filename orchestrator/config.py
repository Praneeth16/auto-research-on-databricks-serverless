"""Configuration for the auto-research orchestrator on Databricks."""

from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    node_type_id: str = "g5.xlarge"  # 1x A10G, 24GB VRAM
    spark_version: str = "15.4.x-gpu-ml-scala2.12"
    num_workers: int = 0  # single-node
    autotermination_minutes: int = 10
    spark_conf: dict = field(default_factory=lambda: {
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]",
    })
    custom_tags: dict = field(default_factory=lambda: {
        "ResourceClass": "SingleNode",
        "project": "auto-research-financial",
    })


@dataclass
class AutoResearchConfig:
    # Databricks workspace
    profile: str = "e2-demo-west"
    host: str = "https://e2-demo-field-eng.cloud.databricks.com"

    # UC Volumes paths
    catalog: str = "main"
    schema: str = "auto_research"
    volume_name: str = "autoresearch"
    volume_base: str = "/Volumes/main/auto_research/autoresearch"
    train_script_path: str = "/Volumes/main/auto_research/autoresearch/train.py"
    data_path: str = "/Volumes/main/auto_research/autoresearch/data"
    adapters_path: str = "/Volumes/main/auto_research/autoresearch/adapters"

    # Training budget
    max_training_seconds: int = 300  # 5 minutes per experiment
    max_experiments: int = 100

    # Agent LLM (Foundation Model API)
    agent_model: str = "databricks-meta-llama-3-1-70b-instruct"
    # Set to use external LLM instead:
    # agent_model: str = "claude-sonnet-4-20250514"
    # agent_provider: str = "anthropic"  # or "openai"

    # MLflow
    experiment_name: str = "/Users/auto-research/financial-lora"

    # Cluster
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    # Local paths (for development/testing)
    local_train_script: str = "train.py"
    local_results: str = "research/results.tsv"
    local_live_dir: str = "research/live"
    local_notes: str = "research/notes.md"
