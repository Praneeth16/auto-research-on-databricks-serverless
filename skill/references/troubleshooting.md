# Troubleshooting Guide

## OOM (Out of Memory) on A10G

**Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory` or process killed by OOM killer.

**Fixes (in order of impact)**:

1. **Enable QLoRA** — switch from FP16/BF16 LoRA to 4-bit quantized base model:
   ```python
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype="bfloat16",
       bnb_4bit_use_double_quant=True,
   )
   ```

2. **Enable gradient checkpointing** — trades compute for VRAM (~30% savings):
   ```python
   training_config = SFTConfig(gradient_checkpointing=True, ...)
   ```

3. **Reduce batch size** — halve `per_device_train_batch_size`, double `gradient_accumulation_steps`:
   ```python
   per_device_train_batch_size=2,   # was 4
   gradient_accumulation_steps=4,   # was 2
   ```

4. **Reduce `max_length`** — shorter sequences use less activation memory:
   ```python
   max_length=512,  # was 1024
   ```

5. **Reduce LoRA rank** — fewer trainable parameters:
   ```python
   r=8,  # was 16
   ```

6. **Use `paged_adamw_8bit` optimizer** — reduces optimizer state memory:
   ```python
   optim="paged_adamw_8bit",
   ```

## Cluster Startup Time Eating Into 5-Minute Budget

**Symptom**: Training job times out before completing because the cluster took 3-4 minutes to start.

**Fixes**:

1. **Use warm pools (cluster policies)**:
   Configure a cluster policy with `"cluster_type": "pool"` and a pre-warmed instance pool:
   ```python
   new_cluster = {
       "instance_pool_id": "your-pool-id",
       "spark_version": "15.4.x-gpu-ml-scala2.12",
       "num_workers": 0,
   }
   ```

2. **Use pre-started clusters**:
   Submit to an already-running cluster instead of creating a new one:
   ```python
   task = SubmitTask(
       task_key="train",
       existing_cluster_id="existing-cluster-id",
       spark_python_task={...},
   )
   ```

3. **Increase timeout for cold starts**:
   If warm pools are unavailable, set `timeout_seconds=600` (10 min) and subtract
   estimated startup time from the training step budget.

4. **Use serverless compute** (if available in your region for GPU workloads):
   Serverless clusters start in seconds, but GPU serverless availability varies.

## Tokenizer Issues with Qwen Models

**Symptom**: `ValueError: Tokenizer class QWenTokenizer does not exist` or model loading hangs.

**Fix**: Always pass `trust_remote_code=True`:
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    trust_remote_code=True,
    ...
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B",
    trust_remote_code=True,
)
```

**Symptom**: Tokenizer does not have a `pad_token`, causing training errors.

**Fix**: Set the pad token to EOS:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## MLflow Logging Overhead

**Symptom**: Training wall time is significantly longer than expected, with pauses between steps.

**Fix**: Reduce logging frequency and use async logging:
```python
training_config = SFTConfig(
    logging_steps=50,           # was 10; less frequent logging
    save_strategy="no",         # disable checkpoint saving during short runs
    eval_strategy="steps",
    eval_steps=100,             # evaluate less often
    ...
)
```

If MLflow artifact logging is slow:
```bash
export MLFLOW_ENABLE_ASYNC_LOGGING=true
```

## Dataset Loading Failures

**Symptom**: `FileNotFoundError` or `PermissionError` when loading from UC Volumes.

**Fixes**:

1. Verify the path format uses `/Volumes/` (capital V):
   ```python
   dataset_path = "/Volumes/catalog/schema/autoresearch/data/train.jsonl"
   ```

2. Check Unity Catalog permissions — the cluster service principal needs `READ VOLUME` on the volume.

3. For JSONL files, ensure each line is valid JSON:
   ```bash
   python -c "import json; [json.loads(l) for l in open('train.jsonl')]"
   ```

## LoRA Adapter Saving Fails

**Symptom**: `PermissionError` or `OSError` when saving adapter to UC Volumes.

**Fix**: The cluster service principal needs `WRITE VOLUME` permission:
```sql
GRANT WRITE VOLUME ON VOLUME catalog.schema.autoresearch TO `service-principal-name`;
```

Also verify the directory exists:
```python
import os
os.makedirs(adapter_save_path, exist_ok=True)
```

## Training Loss Not Decreasing

**Symptom**: `train_loss` stays flat or increases across steps.

**Possible causes and fixes**:

1. **Learning rate too high** — reduce by 10x:
   ```python
   learning_rate=2e-5,  # was 2e-4
   ```

2. **Dataset too small** — fewer than 100 examples can cause unstable training.
   Add more data or increase epochs instead of max_steps.

3. **LoRA rank too low** — increase from 8 to 16 or 32.

4. **Wrong target modules** — ensure you are adapting the correct layers for the model architecture.

## bitsandbytes Installation Issues

**Symptom**: `ImportError: libbitsandbytes_cuda118.so: cannot open shared object file`

**Fix**: On Databricks ML Runtime 15.4+, `bitsandbytes` is pre-installed. If not:
```bash
%pip install bitsandbytes>=0.43.0
```

Ensure the CUDA version matches the runtime. The ML Runtime uses CUDA 12.x,
so `bitsandbytes` must be built for CUDA 12.

## Job Submission Errors

**Symptom**: `databricks.sdk.errors.InvalidParameterValue` when submitting a job.

**Common causes**:

1. **Invalid node type for region** — `g5.xlarge` may not be available in all regions.
   Check available types:
   ```python
   w = WorkspaceClient()
   node_types = w.clusters.list_node_types()
   gpu_types = [nt for nt in node_types.node_types if nt.node_info and nt.node_info.gpu_count > 0]
   ```

2. **Spark version mismatch** — ensure the GPU ML runtime version exists:
   ```python
   versions = w.clusters.spark_versions()
   gpu_versions = [v for v in versions.versions if "gpu-ml" in v.key]
   ```
