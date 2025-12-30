import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _disable_hf_transfer_if_missing() -> None:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") not in {"1", "true", "True"}:
        return
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("hf_transfer not installed; disabling HF_HUB_ENABLE_HF_TRANSFER for this run.")


def resolve_torch_dtype(precision: str) -> torch.dtype | None:
    precision = (precision or "").lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return None


def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: torch.dtype | None = None,
    device: str | None = None,
    local_path: str | None = None,
):
    _disable_hf_transfer_if_missing()
    model_path = local_path or model_name_or_path
    if local_path and not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local model path not found: {local_path}")
    local_only = bool(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_only,
    )
    if device:
        model = model.to(device)
    return model, tokenizer
