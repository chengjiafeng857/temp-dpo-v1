import argparse
import inspect
import os
import random
from datetime import datetime
from typing import Any, cast

import yaml
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_constant_schedule_with_warmup,
    set_seed,
)
import wandb

from dataset_process import _build_sft_records, _is_hh_dataset, _is_shp_dataset
from model_utils import resolve_torch_dtype, load_model_and_tokenizer


def _torch_debug_info(device: str) -> dict:
# Collect torch and cuda debug info for wandb logging
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": device,
        "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        info.update(
            {
                "cuda_device_name": props.name,
                "cuda_device_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_bf16_supported": getattr(torch.cuda, "is_bf16_supported", lambda: False)(),
            }
        )
    return info


def load_yaml_config(path: str) -> dict[str, Any]:
# Load YAML config file
    with open(path, "r") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def random_controler(seed=42):
# Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _sync_model_tokens_with_tokenizer(model, tokenizer) -> None:
# Ensure model config tokens are in sync with tokenizer tokens
    token_keys = ("pad_token_id", "bos_token_id", "eos_token_id")
    for key in token_keys:
        tok_value = getattr(tokenizer, key, None)
        if hasattr(model.config, key):
            setattr(model.config, key, tok_value)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, key):
            setattr(model.generation_config, key, tok_value)


def _load_raw_sft_split(config):
# We rewrite data loading logic here, to acomodate Trainer API, which expects a Dataset object.
# This is a temporary workaround, we can factor a shared “load raw splits” helper in dataset_process.py that returns train_raw/eval_raw from build_sft_train_val.
    dataset_name = config["dataset"]["dataset_name"]
    subset = config["dataset"].get("subset", "train")
    seed = config["dataset"].get("seed", 42)
    eval_ratio = 0.05

    if _is_hh_dataset(dataset_name) or _is_shp_dataset(dataset_name):
        dataset_id = "Anthropic/hh-rlhf" if _is_hh_dataset(dataset_name) else "stanfordnlp/SHP"
        dataset_dict = load_dataset(dataset_id)
        if isinstance(dataset_dict, dict):
            if subset in dataset_dict:
                train_raw = dataset_dict[subset]
            else:
                train_raw = load_dataset(dataset_id, split=subset)
            if "test" in dataset_dict:
                eval_raw = dataset_dict["test"]
            elif "validation" in dataset_dict:
                eval_raw = dataset_dict["validation"]
            else:
                split = train_raw.train_test_split(test_size=eval_ratio, seed=seed)
                train_raw, eval_raw = split["train"], split["test"]
        else:
            split = dataset_dict.train_test_split(test_size=eval_ratio, seed=seed)
            train_raw, eval_raw = split["train"], split["test"]
    else:
        raw = load_dataset(dataset_name, split=subset)
        split = raw.train_test_split(test_size=eval_ratio, seed=seed)
        train_raw, eval_raw = split["train"], split["test"]

    return train_raw, eval_raw, dataset_name


def _build_sft_dataset(raw, dataset_name):
    records = _build_sft_records(raw, dataset_name)
    return Dataset.from_list(records)


def _tokenize_sft_dataset(ds, tokenizer, max_len):
# Tokenize SFT dataset with prompt masking in the labels, padding, and truncation.
# This is a temp simplified version of process_sft_ds in dataset_process.py, adapted for Hugging Face Trainer.
# The function returns a tokenized Dataset ready for training.
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id  #redundant for safe-check
    bos_token_id = tokenizer.bos_token_id

    def _tokenize(example):
        prompt = example["prompt"]
        sft_target = example["sft_target"]
        truncation_mode = example.get("truncation_mode", "keep_start")

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if bos_token_id is not None and (not prompt_ids or prompt_ids[0] != bos_token_id):
            prompt_ids = [bos_token_id] + prompt_ids
        target_ids = tokenizer(sft_target, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > max_len:
            if truncation_mode == "keep_end":
                if bos_token_id is not None and input_ids and input_ids[0] == bos_token_id and max_len > 0:
                    tail_len = max_len - 1
                    input_ids = [bos_token_id] + input_ids[-tail_len:] if tail_len > 0 else [bos_token_id]
                    labels = [-100] + labels[-tail_len:] if tail_len > 0 else [-100]
                else:
                    input_ids = input_ids[-max_len:]
                    labels = labels[-max_len:]
            else:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

        attention_mask = [1] * len(input_ids)
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = ds.map(_tokenize, remove_columns=ds.column_names)
    tokenized.set_format("torch")
    return tokenized


def _debug_sft_samples(raw_ds, tokenizer, max_len, sample_size=5) -> None:
    if sample_size <= 0:
        return
    total = len(raw_ds)
    if total == 0:
        print("SFT debug: no samples to show.")
        return
    sample_size = min(sample_size, total)
    indices = random.sample(range(total), sample_size)
    raw_sample = raw_ds.select(indices)
    tokenized_sample = _tokenize_sft_dataset(raw_sample, tokenizer, max_len)
    print("SFT debug samples (raw + tokenized):")
    for i, idx in enumerate(indices, start=1):
        raw_item = raw_sample[i - 1]
        tokenized_item = {
            key: (value.tolist() if torch.is_tensor(value) else value)
            for key, value in tokenized_sample[i - 1].items()
        }
        print(f"[sample {i}/{sample_size}] index={idx}")
        print("raw:", raw_item)
        print("tokenized:", tokenized_item)


def train_sft(policy, tokenizer, config: dict[str, Any], device: str):
# Main training proccess, using Trainer from Hugging Face Transformers.
    # Ensure model is in training mode, safety check for switching from evaluation mode
    policy.train()
    policy.requires_grad_(True)
    #redundant, already ran in main()
    # _sync_model_tokens_with_tokenizer(policy, tokenizer)
# Dataset preparation
    sft_config = config.get("sft_training", {})
    gradient_checkpointing = sft_config.get("gradient_checkpointing", False)
    if gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        if getattr(policy.config, "use_cache", None) is not None:
            policy.config.use_cache = False

    set_seed(config.get("dataset", {}).get("seed", 42))
    train_raw, eval_raw, dataset_name = _load_raw_sft_split(config)
    train_ds = _build_sft_dataset(train_raw, dataset_name)
    eval_ds = _build_sft_dataset(eval_raw, dataset_name)
# Tokenize datasets
    max_len = config["dataset"]["max_len"]
    # debug samples
    debug_samples = int(sft_config.get("debug_samples", 5))
    debug_show_samples = sft_config.get("debug_show_samples", True)
    if debug_show_samples:
        _debug_sft_samples(train_ds, tokenizer, max_len, sample_size=debug_samples)
    train_ds = _tokenize_sft_dataset(train_ds, tokenizer, max_len)
    eval_ds = _tokenize_sft_dataset(eval_ds, tokenizer, max_len)

# TrainingArguments
    output_dir = sft_config.get("save_dir", "./logs/sft")
    log_steps = max(1, int(sft_config.get("log_steps", 50)))
    eval_steps = sft_config.get("eval_steps", log_steps)
    eval_strategy = sft_config.get("evaluation_strategy") or sft_config.get("eval_strategy") or "steps"
    # warmup_steps = int(sft_config.get("warmup_steps", 150))

    training_kwargs = {
        "output_dir": output_dir,
        # multiplied by the number of GPUs
        "per_device_train_batch_size": sft_config.get("batch_size", 8),
        "per_device_eval_batch_size": sft_config.get("eval_batch_size", 8),
        # Number of batches accumulated before one optimizer step.
        # effective batch size = batch_size * gradient_accumulation_steps
        "gradient_accumulation_steps": sft_config.get("gradient_accumulation_steps", 8),
        # enable "eval_accumulation_steps" for memory-efficient eval if needed
        # "eval_accumulation_steps": sft_config.get("eval_accumulation_steps", 30),
        "dataloader_num_workers": sft_config.get("dataloader_num_workers", 0),
        "num_train_epochs": sft_config.get("epochs", 1),
        "learning_rate": float(sft_config.get("learning_rate", 2e-5)),
        # bf16 for training computes
        "bf16": True,
        # use RMSprop optimizer to align with Beta-DPO setup, recommand using AdamW-8bit for SFT.
        "optim": "adamw_bnb_8bit",
        # constent scheduler is used in Beta-DPO, but cosine is more common for SFT.
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": log_steps,
        "logging_first_step": True,
        "report_to": ["wandb"] if wandb.run is not None else [],
        "gradient_checkpointing": gradient_checkpointing,
    }
    args_params = inspect.signature(TrainingArguments).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in args_params else "eval_strategy"
    training_kwargs[eval_key] = eval_strategy
    training_args = TrainingArguments(**training_kwargs)

    # Optimizer setup, RMSprop optimizer to align with Beta-DPO.
    # optimizer = torch.optim.RMSprop(policy.parameters(), lr=training_args.learning_rate)
    # Match Beta-DPO warmup-to-constant schedule.
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

# Trainer
    trainer_kwargs = {
        "model": policy,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
        # "optimizers": (optimizer, scheduler),
    }
    trainer_params = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    return trainer


def main():
# Main function to run the SFT training process
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_controler()
# wandb init
    wandb.init(
        project=config.get("wandb_project", "dpo-v1"),
        name=config.get("run_name", "sft-run"),
        config=config,
    )
# Load model and tokenizer
    policy_name = config["policy_name"]
    torch_dtype = resolve_torch_dtype(config.get("precision"))
    local_model_path = config.get("local_model_path") or config.get("sft_training", {}).get("local_model_path")
    fix_mistral_regex = config.get("fix_mistral_regex", True)
    policy, tokenizer = load_model_and_tokenizer(
        policy_name,
        torch_dtype=torch_dtype,
        device=device,
        local_path=local_model_path,
        fix_mistral_regex=fix_mistral_regex,
    )
# Set up tokenizer padding, if pad_id not in tokenizer, default to end-of-sequence token 
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id
# Ensure model tokens are in sync with tokenizer tokens
    _sync_model_tokens_with_tokenizer(policy, tokenizer)
# Log torch debug info to wandb
    if wandb.run is not None:
        debug_info = _torch_debug_info(device)
        wandb.config.update(debug_info, allow_val_change=True)
# Run SFT training
    trainer = train_sft(policy, tokenizer, config, device)
# Save model and tokenizer
    base_output_dir = config.get("sft_training", {}).get("save_dir", "./logs/sft")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")

    mount_base_dir = config.get("sft_training", {}).get("mount_save_dir") or config.get("mount_save_dir")
    if mount_base_dir:
        mount_output_dir = os.path.join(mount_base_dir, timestamp)
        os.makedirs(mount_output_dir, exist_ok=True)
        trainer.save_model(mount_output_dir)
        tokenizer.save_pretrained(mount_output_dir)
        print(f"SFT model also saved to {mount_output_dir}")


if __name__ == "__main__":
    main()
