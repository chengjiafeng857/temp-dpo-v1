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
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_process import build_train_val
from dpo_loss_v1 import dpo_loss
from batch_log_prob import compute_batch_log_prob
import wandb
import os
import json
import shutil
import functools
import inspect
import time

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


def _rank_gpu_info(rank, local_rank):
    info = {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        info.update(
            {
                "cuda_device_index": int(device_idx),
                "cuda_device_name": props.name,
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def _gather_rank_objects(obj, is_distributed, world_size):
    if not is_distributed:
        return [obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def log_distributed_info(is_distributed, rank, local_rank, world_size, is_main_process):
    info = _rank_gpu_info(rank, local_rank)
    info_list = _gather_rank_objects(info, is_distributed, world_size)
    if not (is_main_process and wandb.run is not None):
        return

    flat_info = {}
    for item in info_list:
        r = item.get("rank", "unknown")
        flat_info[f"gpu_rank{r}_name"] = item.get("cuda_device_name")
        flat_info[f"gpu_rank{r}_total_mem_gb"] = item.get("cuda_total_memory_gb")
        flat_info[f"gpu_rank{r}_device_index"] = item.get("cuda_device_index")
        flat_info[f"gpu_rank{r}_visible_devices"] = item.get("cuda_visible_devices")
        flat_info[f"gpu_rank{r}_visible_count"] = item.get("cuda_device_count")

    wandb.config.update(
        {
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "distributed": bool(is_distributed),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        allow_val_change=True,
    )
    wandb.config.update(flat_info, allow_val_change=True)


def gather_gpu_memory_stats(is_distributed, world_size, rank):
    if not torch.cuda.is_available():
        return None
    stats = {
        "rank": int(rank),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }
    return _gather_rank_objects(stats, is_distributed, world_size)

# eval()

def init_distributed(enable_fsdp: bool = False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 and not enable_fsdp:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:29500",
                rank=0,
                world_size=1,
            )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
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
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    enable_fsdp = bool(config.get("dpo_training", {}).get("fsdp", False))
    is_distributed, rank, local_rank, world_size = init_distributed(enable_fsdp=enable_fsdp)
    is_main_process = rank == 0
    if is_main_process and enable_fsdp and world_size == 1:
        print("FSDP is enabled with world_size=1. Use torchrun for multi-GPU sharding.")

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
    else:
        device = torch.device("cpu")

    random_controler()

    # initial wandb
    if is_main_process:
        wandb.init(project=config.get('wandb_project','handwritten-dpo'),
                   name=config.get('run_name','run'),
                   config=config)
        wandb.define_metric("train/step")
        wandb.define_metric("loss", step_metric="train/step")
        wandb.define_metric("chosen_rewards", step_metric="train/step")
        wandb.define_metric("rejected_rewards", step_metric="train/step")
        wandb.define_metric("model_margin", step_metric="train/step")
        wandb.define_metric("system/time_s")
        wandb.define_metric("gpu/*", step_metric="system/time_s")
    log_distributed_info(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    # load model and tokenizer
    policy_name = config['policy_name']
    ref_name = config['ref_name']
    policy = AutoModelForCausalLM.from_pretrained(policy_name).to(device)
    tok = AutoTokenizer.from_pretrained(policy_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    policy.config.pad_token_id = tok.pad_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    
    # load dataset
    train_loader, val_loader = build_train_val(config=config, tokenizer=tok)

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

    start_time = time.time()

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
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step = epoch * len(train_loader) + step + 1

            gpu_stats = gather_gpu_memory_stats(
                is_distributed=is_distributed,
                world_size=world_size,
                rank=rank,
            )

            # log the training info
            log_now = (step + 1) % log_steps == 0
            if log_now:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")

            if is_main_process and wandb.run is not None:
                log_payload = {"system/time_s": time.time() - start_time}
                if gpu_stats:
                    for item in gpu_stats:
                        r = item.get("rank", "unknown")
                        log_payload[f"gpu/allocated_gb_rank{r}"] = item.get("allocated_gb")
                        log_payload[f"gpu/reserved_gb_rank{r}"] = item.get("reserved_gb")
                        log_payload[f"gpu/max_allocated_gb_rank{r}"] = item.get("max_allocated_gb")
                        log_payload[f"gpu/max_reserved_gb_rank{r}"] = item.get("max_reserved_gb")
                if log_now:
                    log_payload.update(
                        {
                            "train/step": global_step,
                            'loss': avg_loss,
                            'chosen_rewards': avg_chosen_rewards.item(),
                            'rejected_rewards': avg_rejected_rewards.item(),
                            'model_margin': avg_model_margin.item(),
                        }
                    )
                wandb.log(log_payload, step=global_step)

            if log_now:
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
