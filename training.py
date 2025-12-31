import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from dataset_process import build_train_val
from dpo_loss_v1 import dpo_loss
from batch_log_prob import compute_batch_log_prob
from model_utils import dump_dpo_debug_samples, load_model, load_tokenizer
import wandb
import os
import json
import shutil
import functools
import inspect

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
# transform the batch to the device
def to_device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

# compute and log model margin
def compute_and_log_model_margin(model_margin, epoch_dir, epoch, step, JSONL_PATH):
    # full array
    # using numpy to process, so only on cpu
    m = model_margin.detach().float().cpu().numpy()  
                
    # 1) save full margins as .npy (raw, lossless)
    # step: batch index
    npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
    np.save(npy_path, m)

    # 2) write a readable per-batch record to ONE jsonl file
    # summary stats
    # quantiles
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
                
    record = {
        "epoch": int(epoch),
        "step": int(step),
        "batch_size": int(m.shape[0]),
        "mean": float(m.mean()),
        "std": float(m.std(ddof=0)),
        "min": float(m.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(m.max()),
        "pos_frac": float((m > 0).mean()),
        "npy": npy_path,
        "sample": [float(x) for x in m[:]],
    }

    # save in the jsonl file       
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# control randomness
def random_controler(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

# eval()

def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP multi-GPU training requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    return True, rank, local_rank, world_size


def get_fsdp_auto_wrap_policy(model):
    layer_cls = set()
    if hasattr(model, "_no_split_modules") and model._no_split_modules:
        no_split = set(model._no_split_modules)
        for module in model.modules():
            if module.__class__.__name__ in no_split:
                layer_cls.add(module.__class__)

    if not layer_cls:
        return None

    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)


def wrap_fsdp_model(model, use_bf16, device):
    auto_wrap_policy = get_fsdp_auto_wrap_policy(model)
    mixed_precision = None
    if use_bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": device,
    }

    if "use_orig_params" in inspect.signature(FSDP).parameters:
        fsdp_kwargs["use_orig_params"] = True

    return FSDP(model, **fsdp_kwargs)


def save_model(policy, save_path, is_main_process):
    if isinstance(policy, FSDP):
        state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, state_dict_cfg):
            state_dict = policy.state_dict()
        if is_main_process:
            policy.module.save_pretrained(save_path, state_dict=state_dict)
    else:
        if is_main_process:
            policy.save_pretrained(save_path)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--policy-name", type=str, default=None, help="Override policy model name/path")
    parser.add_argument("--ref-name", type=str, default=None, help="Override ref model name/path")
    args=parser.parse_args()

    config = load_yaml_config(args.config)
    if args.policy_name:
        config["policy_name"] = args.policy_name
    if args.ref_name:
        config["ref_name"] = args.ref_name

    is_distributed, rank, local_rank, world_size = init_distributed()
    is_main_process = rank == 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
    else:
        device = torch.device("cpu")

    random_controler()

    # initial wandb
    if is_main_process:
        wandb.init(project=config.get('wandb_project','dpo-v1'),
                   name=config.get('run_name','run'),
                   config=config)

    # load model and tokenizer
    policy_name = config['policy_name']
    ref_name = config.get('ref_name', policy_name)
    if ref_name != policy_name:
        if is_main_process:
            print(f"Overriding ref_name ({ref_name}) to match policy_name ({policy_name}).")
        ref_name = policy_name
    if is_main_process:
        print(f"Using policy model: {policy_name}")
        print(f"Using ref model: {ref_name}")
    policy = load_model(policy_name, device=device)
    fix_mistral_regex = config.get("fix_mistral_regex", True)
    tok = load_tokenizer(policy_name, fix_mistral_regex=fix_mistral_regex)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    policy.config.pad_token_id = tok.pad_token_id

    ref_model = load_model(ref_name, device=device)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    
    # load dataset
    train_loader, val_loader = build_train_val(config=config, tokenizer=tok)
    dpo_debug = config.get("dpo_debug", {}) or {}
    if is_main_process and dpo_debug.get("enabled", False):
        dump_dpo_debug_samples(
            train_loader,
            seed=int(config.get("dataset", {}).get("seed", 42)),
            max_samples=int(dpo_debug.get("max_samples", 3)),
            output_dir=str(dpo_debug.get("output_dir", "/logs/debugging")),
        )

    # using bf 16
    use_bf16 = config['precision'] == 'bf16'
    if use_bf16:
        policy.to(dtype=torch.bfloat16)
        ref_model.to(dtype=torch.bfloat16)

    if is_distributed:
        policy = wrap_fsdp_model(policy, use_bf16=use_bf16, device=device)
        ref_model = wrap_fsdp_model(ref_model, use_bf16=use_bf16, device=device)

    policy.train()
    ref_model.eval()

    # define optimizer
    optimizer = None
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))
            print("Using bitsandbytes 8-bit AdamW optimizer")
        except ImportError:
            print("bitsandbytes not found.")
        except Exception as e:
            print(f"Failed to initialize 8-bit AdamW: {e}.")

    if optimizer is None:
        print("Falling back to torch.optim.AdamW.")
        optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    # define margin log
    LOG_DIR = "logs/margins"
    os.makedirs(LOG_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")


    # training loop

    # every epoch create a folder to save the model_margin
    epochs = config['dpo_training']['epochs']
    log_steps = config['dpo_training']['log_steps']
    dpo_training_config = config.get("dpo_training", {}) or {}
    gradient_accumulation_steps = int(dpo_training_config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    warmup_steps = int(dpo_training_config.get("warmup_steps", 0))
    grad_clip_enabled = bool(dpo_training_config.get("gradient_clipping_enabled", False))
    max_grad_norm = float(dpo_training_config.get("max_grad_norm", 1.0))
    if grad_clip_enabled and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0 when gradient clipping is enabled")
    steps_per_epoch = (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_steps = epochs * steps_per_epoch
    scheduler = None
    if warmup_steps > 0:
        if warmup_steps > total_steps:
            if is_main_process:
                print(f"Warmup steps ({warmup_steps}) exceed total steps ({total_steps}); clamping.")
            warmup_steps = total_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if is_main_process:
            print(f"Using linear warmup for {warmup_steps} steps.")

    for epoch in range(epochs):
        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(enumerate(train_loader),
                total=len(train_loader),
                desc=f"train | epoch {epoch+1}/{epochs}",
                dynamic_ncols=True, leave=False, disable=not is_main_process)
        
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            # generate logits for each part
            # using bf16
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16): 
                # compute log_prob
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(batch, policy=policy, ref_model=ref_model)

                # compute loss
                loss_raw, chosen_rewards, rejected_rewards, model_margin = dpo_loss(
                     policy_chosen_log_prob=policy_chosen_log_prob,
                     policy_rejected_log_prob=policy_rejected_log_prob,
                     ref_chosen_log_prob=ref_chosen_log_prob,
                     ref_rejected_log_prob=ref_rejected_log_prob,
                     beta=config['dpo_training']['beta']
                     )
                
                if is_main_process:
                    compute_and_log_model_margin(
                        model_margin=model_margin,
                        epoch_dir=epoch_dir,
                        epoch=epoch,
                        step=step,
                        JSONL_PATH=JSONL_PATH
                    )
                
                loss_raw_mean = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()
                preference_accuracy = (chosen_rewards > rejected_rewards).float().mean()
                chosen_token_counts = (batch["chosen_labels"] != -100).sum(-1).clamp(min=1)
                rejected_token_counts = (batch["rejected_labels"] != -100).sum(-1).clamp(min=1)
                per_token_chosen_reward = chosen_rewards / chosen_token_counts
                per_token_rejected_reward = rejected_rewards / rejected_token_counts
                preference_accuracy_per_token = (
                    per_token_chosen_reward > per_token_rejected_reward
                ).float().mean()

            loss = loss_raw_mean / gradient_accumulation_steps
            loss.backward()

            is_accum_step = (step + 1) % gradient_accumulation_steps == 0
            is_last_step = (step + 1) == len(train_loader)
            if is_accum_step or is_last_step:
                if grad_clip_enabled:
                    if hasattr(policy, "clip_grad_norm_"):
                        policy.clip_grad_norm_(max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss_raw_mean.item()

            # log the training info
            if is_main_process and epoch == 0 and step == 0:
                wandb.log({
                    'preference_accuracy': preference_accuracy.item(),
                    'preference_accuracy_per_token': preference_accuracy_per_token.item(),
                }, step=0)
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                if is_main_process:
                    current_lr = optimizer.param_groups[0].get("lr", None)
                    wandb.log({
                        'loss': avg_loss,
                        'chosen_rewards': avg_chosen_rewards.item(),
                        'rejected_rewards': avg_rejected_rewards.item(),
                        'model_margin': avg_model_margin.item(),
                        'preference_accuracy': preference_accuracy.item(),
                        'preference_accuracy_per_token': preference_accuracy_per_token.item(),
                        'lr': current_lr
                    }, step=(epoch * len(train_loader) + step + 1))
                running_loss = 0.0

    # save model
    mount_dir = config.get('mount_dir', None)
    if mount_dir:
        save_path = os.path.join(mount_dir, "dpo_model")
        print(f"Saving model to mount dir: {save_path}")
    else:
        save_path = "dpo_model"
        print(f"Saving model to default path: {save_path}")
    
    if is_distributed:
        dist.barrier()

    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
    if is_distributed:
        dist.barrier()

    save_model(policy, save_path, is_main_process=is_main_process)

    # copy logs to mount dir
    if mount_dir and is_main_process:
        mount_log_dir = os.path.join(mount_dir, "logs")
        print(f"Copying logs to mount dir: {mount_log_dir}")
        try:
            shutil.copytree(LOG_DIR, mount_log_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying logs to mount dir: {e}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
