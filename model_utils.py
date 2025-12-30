import json
import os
import random

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


def _is_local_dir(path: str) -> bool:
    return os.path.isdir(path)

# Load tokenizer with optional fix for Mistral regex issue
def load_tokenizer(
    model_name_or_path: str,
    local_path: str | None = None,
    fix_mistral_regex: bool = True,
):
    _disable_hf_transfer_if_missing()
    model_path = local_path or model_name_or_path
    if local_path and not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local model path not found: {local_path}")
    local_only = bool(local_path) or _is_local_dir(model_path)
    tokenizer_kwargs = {"local_files_only": local_only}
    if fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
        try:
            return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except TypeError:
            tokenizer_kwargs.pop("fix_mistral_regex", None)
    return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

# Load both model and tokenizer with optional Mistral regex fix
def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: torch.dtype | None = None,
    device: str | None = None,
    local_path: str | None = None,
    fix_mistral_regex: bool = True,
):
    _disable_hf_transfer_if_missing()
    model_path = local_path or model_name_or_path
    if local_path and not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local model path not found: {local_path}")
    local_only = bool(local_path) or _is_local_dir(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_only,
    )
    tokenizer = load_tokenizer(
        model_name_or_path,
        local_path=local_path,
        fix_mistral_regex=fix_mistral_regex,
    )
    if device:
        model = model.to(device)
    return model, tokenizer


def load_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype | None = None,
    device: str | None = None,
    local_path: str | None = None,
):
    _disable_hf_transfer_if_missing()
    model_path = local_path or model_name_or_path
    if local_path and not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local model path not found: {local_path}")
    local_only = bool(local_path) or _is_local_dir(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_only,
    )
    if device:
        model = model.to(device)
    return model

#debugging utility for DPO tokenization issues, prints raw and tokenized samples
def dump_dpo_debug_samples(
    train_loader,
    seed: int,
    max_samples: int = 3,
    output_dir: str = "logs/debugging",
) -> None:
    dataset = train_loader.dataset
    total = len(dataset)
    if total <= 0:
        print("DPO debug: dataset is empty.")
        return

    rng = random.Random(seed)
    sample_count = min(max_samples, total)
    indices = rng.sample(range(total), k=sample_count)
    raw_batch = [dataset[i] for i in indices]

    print("DPO debug: raw samples (pre-tokenization)")
    for i, idx in enumerate(indices, start=1):
        print(f"sample_{i}_index: {idx}")
        print(f"raw_record: {raw_batch[i - 1]}")

    tokenized = train_loader.collate_fn(raw_batch)
    debug_records = []
    for i, idx in enumerate(indices, start=1):
        record = {
            "index": int(idx),
            "raw_record": raw_batch[i - 1],
        }
        for key, value in tokenized.items():
            if torch.is_tensor(value):
                record[key] = value[i - 1].tolist()
            else:
                record[key] = value
        debug_records.append(record)

        print(f"DPO debug sample {i}:")
        for key, value in record.items():
            print(f"{key}: {value}")

    debug_path = os.path.join(output_dir, "dpo_tokenization_debug.jsonl")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(debug_path, "a", encoding="utf-8") as handle:
            for record in debug_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"DPO debug: saved to {debug_path}")
    except OSError as exc:
        print(f"DPO debug: failed to write {debug_path}: {exc}")
