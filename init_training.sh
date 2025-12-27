#!/bin/bash

# Exit on error
set -e

# Auto-shutdown logic
POD_ID="${RUNPOD_POD_ID:-}"
cleanup() {
    if [[ -n "$POD_ID" ]]; then
        echo "[cleanup] Training finished (or script exited). Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID" || true
    else
        echo "[cleanup] RUNPOD_POD_ID not set, skipping auto-shutdown."
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
    uv sync
# If pyproject.toml exists but no lock, install from it
elif [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml. Installing..."
    uv pip install .
# Fallback to requirements.txt
elif [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing..."
    uv pip install -r requirements.txt
else
    echo "Error: No dependency file found (uv.lock, pyproject.toml, or requirements.txt)."
    exit 1
fi

# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please export HF_TOKEN='your_token_here' before running this script."
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# 5. Run Training Scripts
echo "Running training.py..."
python training.py --config config_dpo.yaml

echo "Training initialization sequence completed."
