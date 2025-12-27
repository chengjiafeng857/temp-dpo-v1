# Repository Guidelines

## Project Structure
- Root-level Python modules: `training.py` (main loop), `dataset_process.py` (data loading), `dpo_loss_v1.py`, and `batch_log_prob.py`.
- Configuration lives in `config_dpo.yaml`.
- Environment tooling: `pyproject.toml`, `uv.lock`, and `init_training.sh`.
- Outputs: `logs/margins/` (jsonl + npy) and `dpo_model/` or `mounts/output/dpo_model` when `mount_dir` is set.

## Build, Test, and Development Commands
- `./init_training.sh` sets up `.venv`, installs deps with `uv`, logs into Hugging Face + Weights & Biases, then launches training.
- `uv venv` / `uv sync` creates and syncs the environment from `uv.lock`.
- `python training.py --config config_dpo.yaml` runs a training job using the YAML config.

## Coding Style & Naming Conventions
- Python 3.11 with 4-space indentation; use `snake_case` for functions and variables.
- Keep modules focused on one responsibility and drive side effects from `train()` in `training.py`.
- Align config keys in `config_dpo.yaml` with code access patterns (e.g., `dpo_training.learning_rate`).

## Testing Guidelines
- No automated test suite is defined yet.
- For validation, do a short smoke run by temporarily setting `dataset.subset` (e.g., `train[:1%]`) and `dpo_training.epochs: 1` in `config_dpo.yaml`, then run `python training.py --config config_dpo.yaml`.

## Commit & Pull Request Guidelines
- Recent commits use short, lowercase summaries like `update` or `init`; keep messages brief and imperative.
- PRs should describe model/dataset changes, include config diffs, and note any new dependencies.
- Avoid committing generated artifacts such as `logs/` and `dpo_model/`.

## Security & Configuration Notes
- `HF_TOKEN` and `WANDB_API_KEY` are required by `init_training.sh`.
- `RUNPOD_POD_ID` enables auto-shutdown in `init_training.sh`; set `AUTO_SHUTDOWN=false` to skip.
