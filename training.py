import torch
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from preference_datasets import get_batch_iterator
from dpo_loss_v1 import dpo_loss
from batch_log_prob import compute_batch_log_prob
import wandb
import os
import json

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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random_controler()

    # initial wandb
    wandb.init(project=config.get('wandb_project','handwritten-dpo'),
               name=config.get('run_name','run'),
               config=config)

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
    
    # dataset config extraction
    ds_cfg = config['dataset']
    train_cfg = config['dpo_training']

    policy.train()
    ref_model.eval()

    # define optimizer
    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    # using bf 16
    use_bf16 = config['precision'] == 'bf16'
    if use_bf16:
        policy.to(dtype=torch.bfloat16)
        ref_model.to(dtype=torch.bfloat16)

    # define margin log
    LOG_DIR = "logs/margins"
    os.makedirs(LOG_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")


    # training loop
    # Call get_batch_iterator per epoch to maintain per-epoch margin logging
    log_steps = train_cfg['log_steps']
    epochs = train_cfg['epochs']
    
    for epoch in range(epochs):
        # Create epoch directory for margin logging
        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Create iterator for this epoch (n_epochs=1)
        train_iterator = get_batch_iterator(
            names=ds_cfg['names'],
            tokenizer=tok,
            split=ds_cfg.get('split', 'train'),
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            max_length=ds_cfg.get('max_length', 512),
            max_prompt_length=ds_cfg.get('max_prompt_length', 256),
            sft_mode=False,
            n_epochs=1,  # One epoch at a time
            seed=train_cfg.get('seed', 42) + epoch,  # Different seed per epoch for shuffling
            cache_dir=ds_cfg.get('cache_dir', None),
        )
        
        pbar = tqdm(enumerate(train_iterator),
                desc=f"train | epoch {epoch+1}/{epochs}",
                dynamic_ncols=True, leave=False)
        
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
                     beta=train_cfg['beta']
                     )
                
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

            # log the training info
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                wandb.log({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'chosen_rewards': avg_chosen_rewards.item(),
                    'rejected_rewards': avg_rejected_rewards.item(),
                    'model_margin': avg_model_margin.item()
                })
                running_loss = 0.0

    # save model
    policy.save_pretrained("dpo_model")

if __name__ == "__main__":
    train()
