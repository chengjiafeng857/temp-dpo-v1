#!/bin/bash

# Exit on error
set -e

# Auto-shutdown logic
POD_ID="${RUNPOD_POD_ID:-}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
CONFIG_PATH="config/config_dpo.yaml"
DPO_REF_SOURCE="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --noautoshut)
            AUTO_SHUTDOWN=false
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dpo-ref-source)
            DPO_REF_SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--config PATH] [--noautoshut] [--dpo-ref-source auto|sft|config]"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 [--config PATH] [--noautoshut] [--dpo-ref-source auto|sft|config]"
            exit 1
            ;;
    esac
done

if [[ "$DPO_REF_SOURCE" != "auto" && "$DPO_REF_SOURCE" != "sft" && "$DPO_REF_SOURCE" != "config" ]]; then
    echo "Error: Invalid --dpo-ref-source value: $DPO_REF_SOURCE"
    echo "Usage: $0 [--config PATH] [--noautoshut] [--dpo-ref-source auto|sft|config]"
    exit 1
fi

if [[ "$CONFIG_PATH" == "config_dpo.yaml" && ! -f "$CONFIG_PATH" && -f "config/config_dpo.yaml" ]]; then
    CONFIG_PATH="config/config_dpo.yaml"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    AUTO_SHUTDOWN=false
    exit 1
fi
cleanup() {
    if [[ -n "$POD_ID" && "$AUTO_SHUTDOWN" == "true" ]]; then
        echo "[cleanup] Training finished (or script exited). Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID" || true
    else
        echo "[cleanup] Skipping auto-shutdown (POD_ID unset or AUTO_SHUTDOWN=false)."
    fi
}
trap cleanup EXIT

echo "Starting training initialization..."

# 1. Create venv using uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv' using uv..."
    uv venv
else
    echo "Virtual environment '.venv' already exists."
fi

# 2. Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install dependencies using uv
echo "Installing dependencies..."
# If uv.lock exists, use sync (fastest and most reliable)
if [ -f "uv.lock" ]; then
    echo "Found uv.lock. Syncing environment..."
    uv sync || { echo "Error: uv sync failed."; AUTO_SHUTDOWN=false; exit 1; }
# If pyproject.toml exists but no lock, install from it
elif [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml. Installing..."
    uv pip install . || { echo "Error: uv pip install . failed."; AUTO_SHUTDOWN=false; exit 1; }
# Fallback to requirements.txt
elif [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing..."
    uv pip install -r requirements.txt || { echo "Error: uv pip install requirements.txt failed."; AUTO_SHUTDOWN=false; exit 1; }
else
    echo "Error: No dependency file found (uv.lock, pyproject.toml, or requirements.txt)."
    AUTO_SHUTDOWN=false
    exit 1
fi

# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please export HF_TOKEN='your_token_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# 5. WandB Login
echo "Logging in to WandB..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please export WANDB_API_KEY='your_key_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
wandb login "$WANDB_API_KEY"

# 6. Run Training Scripts
echo "Checking SFT training config..."
RUN_SFT=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r") as handle:
    cfg = yaml.safe_load(handle) or {}
sft_cfg = cfg.get("sft_training", {}) or {}
enabled = sft_cfg.get("enabled", True)
print("true" if enabled else "false")
PY
)

if [[ "$RUN_SFT" == "true" ]]; then
    echo "Running sft_training.py..."
    python sft_training.py --config "$CONFIG_PATH" || { echo "Error: sft_training.py failed."; AUTO_SHUTDOWN=false; exit 1; }
else
    echo "Skipping SFT training (sft_training.enabled=false)."
fi

DPO_POLICY_OVERRIDE=""
DPO_REF_OVERRIDE=""
if [[ "$DPO_REF_SOURCE" == "sft" || ( "$DPO_REF_SOURCE" == "auto" && "$RUN_SFT" == "true" ) ]]; then
    DPO_REF_OVERRIDE=$(python - "$CONFIG_PATH" <<'PY'
import os
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r") as handle:
    cfg = yaml.safe_load(handle) or {}
sft_cfg = cfg.get("sft_training", {}) or {}

base_dirs = []
mount_base = sft_cfg.get("mount_save_dir") or cfg.get("mount_save_dir")
if mount_base:
    base_dirs.append(mount_base)
save_dir = sft_cfg.get("save_dir", "./logs/sft")
if save_dir:
    base_dirs.append(save_dir)

latest_path = ""
latest_mtime = -1.0
for base in base_dirs:
    if not os.path.isdir(base):
        continue
    try:
        entries = os.listdir(base)
    except OSError:
        continue
    for name in entries:
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path

print(latest_path)
PY
)

    if [[ -n "$DPO_REF_OVERRIDE" ]]; then
        DPO_POLICY_OVERRIDE="$DPO_REF_OVERRIDE"
        echo "Using SFT model as DPO policy/ref: $DPO_REF_OVERRIDE"
    elif [[ "$DPO_REF_SOURCE" == "sft" ]]; then
        echo "Error: No SFT output found to use as DPO ref."
        AUTO_SHUTDOWN=false
        exit 1
    else
        echo "No SFT output found; using config ref_name."
    fi
fi

echo "Running training.py..."
if [[ -n "$DPO_POLICY_OVERRIDE" || -n "$DPO_REF_OVERRIDE" ]]; then
    DPO_ARGS=()
    if [[ -n "$DPO_POLICY_OVERRIDE" ]]; then
        DPO_ARGS+=(--policy-name "$DPO_POLICY_OVERRIDE")
    fi
    if [[ -n "$DPO_REF_OVERRIDE" ]]; then
        DPO_ARGS+=(--ref-name "$DPO_REF_OVERRIDE")
    fi
    python training.py --config "$CONFIG_PATH" "${DPO_ARGS[@]}"
else
    python training.py --config "$CONFIG_PATH"
fi

echo "Training initialization sequence completed."
