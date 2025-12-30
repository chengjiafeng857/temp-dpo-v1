# dataset_process.py
import torch
import torch.distributed as dist
from collections import defaultdict
from datasets import load_dataset
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


def _extract_texts(item: Dict[str, str]) -> Tuple[str, str, str]:
    """Normalize supported datasets into (prompt, chosen, rejected) text."""
    if "prompt" in item:
        return item["prompt"], item["chosen"], item["rejected"]

    prompt_txt = "Prompt: " + item["question"].strip() + "\n"
    chosen_txt = "Response: " + item["chosen"].strip() + "\n"
    rejected_txt = "Response: " + item["rejected"].strip() + "\n"
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
            prompt_txt, chosen_txt, _ = _extract_texts(row)
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


def process_ds(batch, tokenizer, max_len: int):
    """
    Build a batch in a β-DPO-like format that matches:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length (optional)

    Labels are masked with -100 on prompt tokens so that dpo_loss_v1.compute_log_prob
    only counts response tokens.
    """
    chosen_texts, rejected_texts = [], []
    prompt_lengths = []

    # Dataset expects fields: question, chosen, rejected (your current code uses these)
    for item in batch:
        question = str(_safe_get(item, "question")).strip()
        chosen = str(_safe_get(item, "chosen")).strip()
        rejected = str(_safe_get(item, "rejected")).strip()

        prompt_txt = f"Prompt: {question}\n"
        chosen_txt = f"Response: {chosen}\n"
        rejected_txt = f"Response: {rejected}\n"

        # prompt length WITHOUT special tokens (β-DPO style)
        prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_ids))

        chosen_texts.append(prompt_txt + chosen_txt)
        rejected_texts.append(prompt_txt + rejected_txt)

    enc_chosen = tokenizer(
        chosen_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    enc_rejected = tokenizer(
        rejected_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # BOS shift: if tokenizer inserts a BOS at position 0, prompt_length should include it
    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.shape[1] > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0

    prompt_length = torch.tensor([pl + bos_shift for pl in prompt_lengths], dtype=torch.long)

    # Build labels: input_ids with prompt tokens masked to -100
    chosen_labels = enc_chosen.input_ids.clone()
    rejected_labels = enc_rejected.input_ids.clone()

    # Clamp prompt_length to max_len to avoid edge issues after truncation
    max_seq_len = chosen_labels.shape[1]
    for i, pl in enumerate(prompt_length.tolist()):
        pl = min(pl, max_seq_len)
        chosen_labels[i, :pl] = -100
        rejected_labels[i, :pl] = -100

    chosen_labels[enc_chosen.attention_mask == 0] = -100
    rejected_labels[enc_rejected.attention_mask == 0] = -100

    return {
        "chosen_input_ids": enc_chosen.input_ids,
        "chosen_attention_mask": enc_chosen.attention_mask,
        "chosen_labels": chosen_labels,

        "rejected_input_ids": enc_rejected.input_ids,
        "rejected_attention_mask": enc_rejected.attention_mask,
        "rejected_labels": rejected_labels,

        # optional: useful for debugging / analysis
        "prompt_length": prompt_length,
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

    # Load a single split (common when dataset only provides "train")
    ds = load_dataset(dataset_name, split=subset)

    # Always create a val split from the loaded split for robustness
    split_ds = ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]

    ds_collate = partial(process_ds, tokenizer=tokenizer, max_len=max_len)

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
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    return train_loader, val_loader
