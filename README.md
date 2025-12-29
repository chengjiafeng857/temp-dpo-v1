# temp-dpo-v1

Minimal DPO training loop for causal LMs with optional FSDP, WandB logging, and simple dataset processing.

## Project layout

- `training.py`: main DPO training loop
- `dataset_process.py`: dataset loading + collation
- `dpo_loss_v1.py`: DPO loss + log-prob helpers
- `batch_log_prob.py`: batch log-prob computation
- `config_dpo.yaml`: training configuration
- `init_training.sh`: environment setup + training entrypoint
- `logs/margins/`: per-batch margin stats (jsonl + npy)
- `dpo_model/` or `mounts/output/dpo_model`: saved model output

## Requirements

- Python 3.11
- CUDA-capable GPU for training
- `HF_TOKEN` and `WANDB_API_KEY` in your environment

## Quick start

```bash
./init_training.sh
```

This will:
- create `.venv` with `uv`
- install deps from `uv.lock`
- log into Hugging Face and Weights & Biases
- run `training.py` with `config_dpo.yaml`

## Run training directly

```bash
uv run python training.py --config config_dpo.yaml
```

### Multi-GPU with FSDP

Enable FSDP in `config_dpo.yaml`:

```yaml
dpo_training:
  fsdp: true
```

Launch with torchrun (example uses 2 GPUs):

```bash
uv run torchrun --nproc_per_node=2 training.py --config config_dpo.yaml
```

## Configuration

All training config lives in `config_dpo.yaml`. Key fields:

- `policy_name` / `ref_name`: model IDs or local paths
- `precision`: `bf16` or `fp32`
- `dataset.dataset_name`: HF dataset name (e.g., `Intel/orca_dpo_pairs`)
- `dataset.max_len`: max sequence length
- `dpo_training.batch_size`, `epochs`, `learning_rate`, `beta`, `log_steps`
- `dpo_training.fsdp`: enable FSDP wrapping
- `mount_dir`: optional output mount (e.g., `mounts/output`)

## Logging

Training logs are sent to W&B from rank 0. When distributed, per-rank GPU details and memory stats are logged so all GPU usage appears in a single run.

Margin stats are saved under `logs/margins/`:
- `logs/margins/margins_log.jsonl`: per-batch summary
- `logs/margins/epoch_XXX/step_XXXXX.npy`: full margin arrays

## Outputs

Model checkpoints are saved to:
- `dpo_model/` by default
- `${mount_dir}/dpo_model` when `mount_dir` is set

Logs are copied to `${mount_dir}/logs` when `mount_dir` is set.

## Troubleshooting

- CUDA OOM: reduce `dpo_training.batch_size` or `dataset.max_len`, or use FSDP with `torchrun`.
- FSDP on single GPU: runs but does not shard; use `torchrun --nproc_per_node` for sharding.

## Notes

- No automated test suite yet. For a smoke run, set `dataset.subset` to a small slice and `dpo_training.epochs: 1`.
