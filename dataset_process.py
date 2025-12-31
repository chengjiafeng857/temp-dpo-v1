# dataset_process.py
import torch
import torch.distributed as dist
from collections import defaultdict
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
from typing import Dict, Iterable, List, Tuple
import tqdm


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the Anthropic prompt from a prompt + response string."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    if search_term_idx == -1:
        raise ValueError(f"Prompt does not contain '{search_term}'")
    return prompt_and_response[:search_term_idx + len(search_term)]


def _is_hh_dataset(name: str) -> bool:
    """Return True when the dataset name refers to the HH dataset."""
    name = name.lower()
    return name in {"hh", "hh-rlhf", "anthropic/hh-rlhf"}


def _is_shp_dataset(name: str) -> bool:
    """Return True when the dataset name refers to the SHP dataset."""
    name = name.lower()
    return name in {"shp", "stanfordnlp/shp"}


def _flatten_prompt_pairs(data: Dict[str, Dict[str, List]]) -> List[Dict[str, str]]:
    """Convert prompt-keyed responses/pairs into flat prompt/chosen/rejected rows."""
    pairs: List[Dict[str, str]] = []
    for prompt, info in data.items():
        responses = info["responses"]
        for chosen_idx, rejected_idx in info["pairs"]:
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": responses[chosen_idx],
                    "rejected": responses[rejected_idx],
                }
            )
    return pairs


def _build_hh_data(
    dataset: Iterable[Dict[str, str]],
    *,
    silent: bool = True,
) -> Dict[str, Dict[str, List]]:
    """Convert HH rows into a prompt-keyed structure."""
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        prompt = extract_anthropic_prompt(row["chosen"])
        chosen_response = row["chosen"][len(prompt):]
        rejected_response = row["rejected"][len(prompt):]

        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend([chosen_response, rejected_response])
        data[prompt]["sft_target"] = chosen_response

    if not silent:
        print(f"Successfully processed {len(data)} prompts.")

    return data


def _build_hh_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return HH preference pairs in flat prompt/chosen/rejected format."""
    return _flatten_prompt_pairs(_build_hh_data(dataset))


def _build_shp_data(dataset: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, List]]:
    """Convert SHP rows into prompt-keyed responses/pairs and sft_target."""
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        responses = [" " + row["human_ref_A"], " " + row["human_ref_B"]]
        scores = [row["score_A"], row["score_B"]]

        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        n_responses = len(data[prompt]["responses"])
        chosen_first = row["labels"] == 1
        data[prompt]["pairs"].append(
            (n_responses, n_responses + 1) if chosen_first else (n_responses + 1, n_responses)
        )
        data[prompt]["responses"].extend(responses)
        data[prompt]["scores"].extend(scores)

    for prompt, info in data.items():
        scores = info["scores"]
        responses = info["responses"]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        info["sft_target"] = responses[best_idx]
        del info["scores"]

    return data


def _build_shp_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert SHP rows into prompt/chosen/rejected pairs with score filtering."""
    return _flatten_prompt_pairs(_build_shp_data(dataset))


def _normalize_dataset(item: Dict[str, str]) -> Tuple[str, str, str]:
    """Normalize supported datasets into (prompt, chosen, rejected) text."""
    if "prompt" in item:
        prompt = str(_safe_get(item, "prompt"))
        chosen = str(_safe_get(item, "chosen"))
        rejected = str(_safe_get(item, "rejected"))
        return prompt, chosen, rejected

    question = str(_safe_get(item, "question")).strip()
    chosen = str(_safe_get(item, "chosen")).strip()
    rejected = str(_safe_get(item, "rejected")).strip()
    prompt_txt = f"Prompt: {question}\n"
    chosen_txt = f"Response: {chosen}\n"
    rejected_txt = f"Response: {rejected}\n"
    return prompt_txt, chosen_txt, rejected_txt


def _build_sft_records(dataset: Iterable[Dict[str, str]], dataset_name: str) -> List[Dict[str, str]]:
    """Build prompt/sft_target records for SFT training."""
    records: List[Dict[str, str]] = []

    if _is_hh_dataset(dataset_name):
        data = _build_hh_data(dataset)
        for prompt, info in data.items():
            records.append(
                {
                    "prompt": prompt,
                    "sft_target": info["sft_target"],
                    "truncation_mode": "keep_end",
                }
            )
        return records

    if _is_shp_dataset(dataset_name):
        data = _build_shp_data(dataset)
        for prompt, info in data.items():
            records.append(
                {
                    "prompt": prompt,
                    "sft_target": info["sft_target"],
                    "truncation_mode": "keep_start",
                }
            )
        return records

    for row in dataset:
        if "prompt" in row and "sft_target" in row:
            prompt_txt = row["prompt"]
            sft_target = row["sft_target"]
        elif "prompt" in row and "chosen" in row:
            prompt_txt = row["prompt"]
            sft_target = row["chosen"]
        else:
            prompt_txt, chosen_txt, _ = _normalize_dataset(row)
            sft_target = chosen_txt
        records.append(
            {
                "prompt": prompt_txt,
                "sft_target": sft_target,
                "truncation_mode": "keep_start",
            }
        )

    return records


def _safe_get(example: dict, key: str):
    if key not in example:
        raise KeyError(
            f"Missing key '{key}' in dataset example. "
            f"Available keys: {list(example.keys())}"
        )
    return example[key]


def process_ds(batch, tokenizer, max_len: int, max_prompt_length: int, truncation_mode: str):
    """
    Build a batch in a β-DPO-like format that matches:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length (optional)

    Labels are masked with -100 on prompt tokens so that dpo_loss_v1.compute_log_prob
    only counts response tokens.
    """
    chosen_input_ids_list = []
    chosen_attention_masks = []
    chosen_labels_list = []
    rejected_input_ids_list = []
    rejected_attention_masks = []
    rejected_labels_list = []
    prompt_lengths = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id
    max_prompt_length = min(int(max_prompt_length), int(max_len))
    add_bos = bos_token_id is not None and max_prompt_length > 0
    prompt_budget = max_prompt_length - 1 if add_bos else max_prompt_length

    for item in batch:
        prompt_txt, chosen_txt, rejected_txt = _normalize_dataset(item)

        prompt_tokens = tokenizer(prompt_txt, add_special_tokens=False)
        chosen_tokens = tokenizer(chosen_txt, add_special_tokens=False)
        rejected_tokens = tokenizer(rejected_txt, add_special_tokens=False)

        prompt_ids = prompt_tokens["input_ids"]
        chosen_ids = chosen_tokens["input_ids"]
        rejected_ids = rejected_tokens["input_ids"]

        if eos_token_id is not None:
            if not chosen_ids or chosen_ids[-1] != eos_token_id:
                chosen_ids = chosen_ids + [eos_token_id]
            if not rejected_ids or rejected_ids[-1] != eos_token_id:
                rejected_ids = rejected_ids + [eos_token_id]

        longer_response_len = max(len(chosen_ids), len(rejected_ids))
        prompt_len_with_bos = len(prompt_ids) + (1 if add_bos else 0)

        if prompt_len_with_bos + longer_response_len > max_len:
            if truncation_mode == "keep_end":
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            elif truncation_mode == "keep_start":
                prompt_ids = prompt_ids[:prompt_budget]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

        prompt_len_with_bos = len(prompt_ids) + (1 if add_bos else 0)
        if prompt_len_with_bos + longer_response_len > max_len:
            response_max_len = max(0, max_len - max_prompt_length)
            chosen_ids = chosen_ids[:response_max_len]
            rejected_ids = rejected_ids[:response_max_len]

        if add_bos and (not prompt_ids or prompt_ids[0] != bos_token_id):
            prompt_ids = [bos_token_id] + prompt_ids

        prompt_lengths.append(len(prompt_ids))

        chosen_seq = prompt_ids + chosen_ids
        rejected_seq = prompt_ids + rejected_ids
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids

        chosen_pad_len = max_len - len(chosen_seq)
        rejected_pad_len = max_len - len(rejected_seq)

        if chosen_pad_len > 0:
            chosen_seq = chosen_seq + [pad_token_id] * chosen_pad_len
            chosen_labels = chosen_labels + [-100] * chosen_pad_len
            chosen_mask = [1] * (max_len - chosen_pad_len) + [0] * chosen_pad_len
        else:
            chosen_mask = [1] * max_len

        if rejected_pad_len > 0:
            rejected_seq = rejected_seq + [pad_token_id] * rejected_pad_len
            rejected_labels = rejected_labels + [-100] * rejected_pad_len
            rejected_mask = [1] * (max_len - rejected_pad_len) + [0] * rejected_pad_len
        else:
            rejected_mask = [1] * max_len

        chosen_input_ids_list.append(torch.tensor(chosen_seq, dtype=torch.long))
        chosen_attention_masks.append(torch.tensor(chosen_mask, dtype=torch.long))
        chosen_labels_list.append(torch.tensor(chosen_labels, dtype=torch.long))
        rejected_input_ids_list.append(torch.tensor(rejected_seq, dtype=torch.long))
        rejected_attention_masks.append(torch.tensor(rejected_mask, dtype=torch.long))
        rejected_labels_list.append(torch.tensor(rejected_labels, dtype=torch.long))

    return {
        "chosen_input_ids": torch.stack(chosen_input_ids_list),
        "chosen_attention_mask": torch.stack(chosen_attention_masks),
        "chosen_labels": torch.stack(chosen_labels_list),

        "rejected_input_ids": torch.stack(rejected_input_ids_list),
        "rejected_attention_mask": torch.stack(rejected_attention_masks),
        "rejected_labels": torch.stack(rejected_labels_list),

        # optional: useful for debugging / analysis
        "prompt_length": torch.tensor(prompt_lengths, dtype=torch.long),
    }


def build_train_val(config, tokenizer):
    """
    Uses config fields:
      dataset.dataset_name
      dataset.subset         (e.g., "train[:90%]")
      dataset.val_ratio
      dataset.seed
      dataset.max_len
      dpo_training.batch_size
    """
    dataset_name = config["dataset"]["dataset_name"]
    subset = config["dataset"].get("subset", "train")
    val_ratio = float(config["dataset"].get("val_ratio", 0.1))
    seed = int(config["dataset"].get("seed", 42))
    max_len = int(config["dataset"]["max_len"])
    batch_size = int(config["dpo_training"]["batch_size"])
    eval_batch_size = int(config["dpo_training"].get("eval_batch_size", batch_size))

    if _is_hh_dataset(dataset_name) or _is_shp_dataset(dataset_name):
        dataset_id = "Anthropic/hh-rlhf" if _is_hh_dataset(dataset_name) else "stanfordnlp/SHP"
        dataset_dict = load_dataset(dataset_id)

        if isinstance(dataset_dict, dict):
            if subset in dataset_dict:
                train_raw = dataset_dict[subset]
            else:
                train_raw = load_dataset(dataset_id, split=subset)

            if "validation" in dataset_dict:
                val_raw = dataset_dict["validation"]
            elif "test" in dataset_dict:
                val_raw = dataset_dict["test"]
            else:
                split_ds = train_raw.train_test_split(test_size=val_ratio, seed=seed)
                train_raw, val_raw = split_ds["train"], split_ds["test"]
        else:
            split_ds = dataset_dict.train_test_split(test_size=val_ratio, seed=seed)
            train_raw, val_raw = split_ds["train"], split_ds["test"]

        if _is_hh_dataset(dataset_name):
            train_ds_raw = Dataset.from_list(_build_hh_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_hh_pairs(val_raw))
        else:
            train_ds_raw = Dataset.from_list(_build_shp_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_shp_pairs(val_raw))
    else:
        # Load a single split (common when dataset only provides "train")
        ds = load_dataset(dataset_name, split=subset)

        # Always create a val split from the loaded split for robustness
        split_ds = ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]

    max_prompt_length = int(config["dataset"].get("max_prompt_length", max_len))
    if _is_hh_dataset(dataset_name):
        truncation_mode = "keep_end"
    else:
        truncation_mode = "keep_start"

    ds_collate = partial(
        process_ds,
        tokenizer=tokenizer,
        max_len=max_len,
        max_prompt_length=max_prompt_length,
        truncation_mode=truncation_mode,
    )

    is_distributed = dist.is_available() and dist.is_initialized()
    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_ds_raw, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_ds_raw, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=not is_distributed,
        sampler=train_sampler,
        collate_fn=ds_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    return train_loader, val_loader
