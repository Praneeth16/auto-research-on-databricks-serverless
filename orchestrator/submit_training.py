"""Submit LoRA fine-tuning jobs to Databricks GPU clusters."""

import io
import time
import json
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.jobs import (
    RunLifeCycleState,
    SubmitTask,
    PythonWheelTask,
    SparkPythonTask,
    NotebookTask,
    ClusterSpec,
    NewCluster,
    SubmitRun,
)


def create_gpu_cluster_spec(config) -> NewCluster:
    """Build a single-node GPU cluster spec from config."""
    return NewCluster(
        spark_version=config.cluster.spark_version,
        node_type_id=config.cluster.node_type_id,
        num_workers=config.cluster.num_workers,
        spark_conf=config.cluster.spark_conf,
        custom_tags=config.cluster.custom_tags,
    )


def submit_training_run(
    client: WorkspaceClient,
    config,
    experiment_id: str,
    experiment_description: str,
) -> dict:
    """
    Submit a training run as a one-time Databricks job.

    Returns dict with:
        - run_id: Databricks run ID
        - status: final status
        - val_loss: parsed from output (or None if failed)
        - duration_seconds: wall clock time
        - output: raw stdout
    """
    # Upload the current train.py to UC Volumes
    train_script_local = Path(config.local_train_script)
    if not train_script_local.exists():
        raise FileNotFoundError(f"train.py not found at {train_script_local}")

    train_script_content = train_script_local.read_text()

    # Upload to volumes via Files API
    volume_path = config.train_script_path
    client.files.upload(
        file_path=volume_path,
        contents=io.BytesIO(train_script_content.encode("utf-8")),
        overwrite=True,
    )

    # Submit the run
    run = client.jobs.submit(
        run_name=f"auto-research-{experiment_id}",
        tasks=[
            SubmitTask(
                task_key="train",
                new_cluster=create_gpu_cluster_spec(config),
                spark_python_task=SparkPythonTask(
                    python_file=volume_path,  # UC Volumes paths work directly, no dbfs: prefix
                    parameters=[
                        "--data-path", config.data_path,
                        "--output-dir", config.adapters_path,
                        "--experiment-id", experiment_id,
                        "--max-seconds", str(config.max_training_seconds),
                        "--results-file", f"{config.volume_base}/val_loss_{experiment_id}.txt",
                    ],
                ),
                timeout_seconds=config.max_training_seconds + 600,  # 10 min buffer for GPU cluster startup
            )
        ],
    )

    run_id = run.run_id
    start_time = time.time()

    # Poll until completion
    while True:
        run_status = client.jobs.get_run(run_id)
        state = run_status.state

        if state.life_cycle_state in (
            RunLifeCycleState.TERMINATED,
            RunLifeCycleState.SKIPPED,
            RunLifeCycleState.INTERNAL_ERROR,
        ):
            break

        time.sleep(15)

    duration = time.time() - start_time

    # Get val_loss from the results file written to UC Volumes by train.py.
    # SparkPythonTask stdout is not reliably accessible, so train.py writes
    # val_loss to a known file path that we read back here.
    output = ""
    val_loss = None

    results_file = f"{config.volume_base}/val_loss_{experiment_id}.txt"
    try:
        resp = client.files.download(results_file)
        content = resp.contents.read().decode("utf-8").strip()
        output = content
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("val_loss:"):
                val_loss = float(line.split(":", 1)[1].strip())
    except Exception as e:
        print(f"Could not read results file {results_file}: {e}")

    # Fallback: try job run logs
    if val_loss is None:
        try:
            run_output = client.jobs.get_run_output(run_id)
            if run_output.logs:
                output = run_output.logs
                for line in output.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("val_loss:"):
                        val_loss = float(line.split(":", 1)[1].strip())
        except Exception:
            pass

    result_state = state.result_state.value if state.result_state else "UNKNOWN"

    return {
        "run_id": run_id,
        "status": result_state,
        "val_loss": val_loss,
        "duration_seconds": round(duration, 1),
        "output": output,
        "experiment_id": experiment_id,
        "description": experiment_description,
    }


def setup_uc_volumes(client: WorkspaceClient, config):
    """Create UC catalog, schema, and volume if they don't exist."""
    try:
        client.catalogs.get(config.catalog)
    except Exception:
        print(f"Catalog '{config.catalog}' not found. Please create it first.")
        raise

    try:
        client.schemas.get(f"{config.catalog}.{config.schema}")
    except Exception:
        print(f"Creating schema {config.catalog}.{config.schema}")
        client.schemas.create(
            name=config.schema,
            catalog_name=config.catalog,
        )

    try:
        client.volumes.read(f"{config.catalog}.{config.schema}.{config.volume_name}")
    except Exception:
        print(f"Creating volume {config.catalog}.{config.schema}.{config.volume_name}")
        client.volumes.create(
            catalog_name=config.catalog,
            schema_name=config.schema,
            name=config.volume_name,
            volume_type=VolumeType.MANAGED,
        )

    # Create subdirectories
    for subdir in ["data", "adapters", "scripts"]:
        try:
            client.files.create_directory(f"{config.volume_base}/{subdir}")
        except Exception:
            pass  # directory may already exist

    print(f"UC Volumes ready at {config.volume_base}")
