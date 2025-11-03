#%%
import os, math, time, json, random
import sys
sys.path.append('/mnt/hdd/Shervin/Thesis')
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from prototype1.model import FlashSDPA, EncoderBlock, CrossAttentionBlock
from prototype1.dataset import GeneMLMDataset
from prototype1.constants import VOCAB_SIZE, NUM_GENE
import pickle
# %%
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int | None = None
    clip_grad_norm: float = 1.0
    log_every: int = 50
    val_every: int = 1000
    ckpt_dir: str = 'checkpoints'
    ckpt_every: int = 2000
    early_stop_patience: int = 10
    amp_prefer_bf16: bool = True
    use_fp16_if_no_bf16: bool = True
    allow_tf32: bool = True
    compile_model: bool = False
    seed: int = 42
    dropout: float = 0.1
    mlp_ratio: float = 4.0
# %%
cfg = TrainConfig()
# %%
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)
# %%
if cfg.allow_tf32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
class GeneEncoder(nn.Module):
    def __init__(self, num_genes=NUM_GENE, load_g2v=True, vocab_size = VOCAB_SIZE, depth=3, d_model=200, n_heads=10, dropout=0.0, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        if load_g2v:
            with open('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/gene_embeddings.pkl','rb') as f:
                gene_embeddings = torch.from_numpy(pickle.load(f))
                gene_embeddings = torch.concat([torch.zeros(1,gene_embeddings.shape[1]), gene_embeddings], dim=0)
            self.g2v_embedding = nn.Embedding.from_pretrained(gene_embeddings, freeze=True)
        else:
            self.g2v_embedding = nn.Embedding(num_genes, d_model)
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.depth = depth
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout, mlp_ratio, activation) for i in range(depth)])

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.vocab_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids):
        x = self.vocab_embedding(input_ids)
        x += self.g2v_embedding.weight
        out = x
        for layer in self.layers:
            out = layer(out)
        
        logits = self.head(out)
        return out
#%%
def is_norm_or_bias(n, p):
    return p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower() or "ln" in n.lower()
#%%
def build_optimizer(model: nn.Module):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        (no_decay if is_norm_or_bias(n, p) else decay).append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)

optimizer = None
# %%
class WarmupCosine:
    def __init__(self, optimizer, warmup, total):
        self.optim = optimizer
        self.warmup = max(1, warmup)
        self.total = max(self.warmup+1, total)
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / self.warmup
        else:
            t = (self.step_num - self.warmup) / (self.total - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * t))
        for g in self.optim.param_groups:
            g["lr"] = cfg.lr * scale
    def get_last_lr(self):
        return [g["lr"] for g in self.optim.param_groups]
# %%
amp_dtype = None
if cfg.amp_prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
elif cfg.use_fp16_if_no_bf16 and torch.cuda.is_available():
    amp_dtype = torch.float16
else:
    amp_dtype = None  # fp32

use_scaler = (amp_dtype == torch.float16)  # GradScaler for fp16 only
scaler = GradScaler(enabled=use_scaler)
# %%
def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: [B, L, V], labels: [B, L] with -100 ignored
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        if mask.sum() == 0:
            return float("nan")
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total
# %%
def save_ckpt(path, model, optimizer, scheduler, step, epoch, best_val, scaler, train_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": {"step_num": scheduler.step_num} if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None and use_scaler else None,
        "step": step, "epoch": epoch, "best_val": best_val,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "train_state": train_state,
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)
# %%
def load_ckpt(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.step_num = ckpt["scheduler"]["step_num"]
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if "rng" in ckpt:
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("train_state", {})
# %%
def evaluate(model, val_loader, loss_fn):
    model.eval()
    tot_loss, tot_acc, n_batches = 0.0, 0.0, 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True).long()
        labels    = batch["labels"].to(device, non_blocking=True).long()
        with autocast(dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False):
            logits = model(input_ids)                 # [B, L, V]
            loss   = loss_fn(logits.transpose(1,2).float(), labels)
        acc = masked_token_accuracy(logits, labels)
        tot_loss += loss.item()
        tot_acc  += (0.0 if np.isnan(acc) else acc)
        n_batches += 1
    return tot_loss / max(1,n_batches), tot_acc / max(1,n_batches)
# %%
def train(model, train_loader, val_loader=None):
    global optimizer
    model.to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # speeds up in PyTorch 2.x

    optimizer = build_optimizer(model)
    # Estimate total steps
    steps_per_epoch = math.ceil(len(train_loader.dataset) / cfg.batch_size)
    total_steps = cfg.max_steps or (cfg.epochs * steps_per_epoch)
    scheduler = WarmupCosine(optimizer, cfg.warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Resume if exists
    best_val, step, start_epoch, bad_counts = float("inf"), 0, 0, 0
    last_ckpt = os.path.join(cfg.ckpt_dir, "last.pt")
    best_ckpt = os.path.join(cfg.ckpt_dir, "best.pt")
    if os.path.exists(last_ckpt):
        step, start_epoch, best_val, _ = load_ckpt(last_ckpt, model, optimizer, scheduler, scaler)
        print(f"Resumed from {last_ckpt} at step={step}, epoch={start_epoch}, best_val={best_val:.4f}")

    model.train()
    running_loss = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True).long()
            labels    = batch["labels"].to(device, non_blocking=True).long()

            with autocast(dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False):
                logits = model(input_ids)                         # [B, L, V]
                loss = loss_fn(logits.transpose(1,2).float(), labels)
                loss = loss / cfg.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum_steps == 0:
                if cfg.clip_grad_norm is not None:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * cfg.grad_accum_steps

            # Logging
            if step % cfg.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{running_loss/cfg.log_every:.4f}", lr=f"{current_lr:.2e}")
                running_loss = 0.0

            # Validation
            if val_loader is not None and step % cfg.val_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, loss_fn)
                print(f"\n[Val] step={step} loss={val_loss:.4f} acc={val_acc:.4f} ppl={math.exp(val_loss):.2f}")
                # Early stopping & best ckpt
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    save_ckpt(best_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})
                    bad_counts = 0
                else:
                    bad_counts += 1
                    if bad_counts >= cfg.early_stop_patience:
                        print("Early stopping.")
                        save_ckpt(last_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})
                        return
                # Always save last after eval
                save_ckpt(last_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})

            # Periodic checkpoint
            if step % cfg.ckpt_every == 0:
                save_ckpt(last_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})

            # Optional step-based stop
            if cfg.max_steps and step >= cfg.max_steps:
                print("Reached max_steps.")
                save_ckpt(last_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})
                return

        # End of epoch checkpoint
        save_ckpt(last_ckpt, model, optimizer, scheduler, step, epoch, best_val, scaler, {"bad_counts": bad_counts})
# %%
model = GeneEncoder(dropout=cfg.dropout,mlp_ratio=cfg.mlp_ratio)
# %%
train_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/train.pkl')
test_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/test.pkl')

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# %%
train(model, train_loader, test_loader)
# %%
