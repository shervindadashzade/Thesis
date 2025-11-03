#%%
import sys
sys.path.append('/mnt/hdd/Shervin/Thesis')
import torch
import torch.nn as nn
import torch.optim as optim
from prototype1.dataset import GeneMLMDataset
from prototype1.model import FlashSDPA, EncoderBlock, CrossAttentionBlock
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from prototype1.constants import NUM_GENE, VOCAB_SIZE
import os
from dataclasses import dataclass
import pickle
from torch.amp import autocast
import math
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gradient_reversal import GradientReversalLayer
import torch.nn.functional as F
#%%
BATCH_SIZE = 8
NUM_WORKERS = 4
LR=2e-4
WEIGHT_DECAY= 0.01
EPOCHS=100
# LOG_EVERY_STEP= 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_GRAD_NORM = 1.0
GRAD_ACCUM_STEPS = 32
WARMUP_PCT = 0.05
FINAL_LR = 1e-6
NAME = 'first_deconfounding'
SEED = 42
EVAL_EVERY_STEP = 23
#%%
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

checkpoint_dir= os.path.join('/mnt/hdd/Shervin/Thesis/prototype1/runs',NAME)
log_dir = os.path.join(checkpoint_dir,'log')
os.makedirs(checkpoint_dir, exist_ok=True)
tensorboard_writer = SummaryWriter(checkpoint_dir)
# %%
train_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/train.pkl')
test_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/test.pkl')
# %%
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
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

        self.exp_head = nn.Linear(d_model, vocab_size, bias=False)
        # self.exp_head.weight = self.vocab_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.shared_head = nn.Linear(d_model, d_model//2, bias=False)
        self.private_head = nn.Linear(d_model, d_model//2, bias=False)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(alpha=1),
            nn.Linear(d_model//2, d_model),
            activation(),
            nn.Linear(d_model, 3)
        )
        # self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, exp_labels=None, domain_labels=None):
        x = self.vocab_embedding(input_ids)
        x += self.g2v_embedding.weight
        out = x
        for layer in self.layers:
            out = layer(out)
        
        # out = self.final_norm(out)

        shared_features = self.shared_head(out)
        private_features = self.private_head(out)

        domain_logits = self.domain_classifier(shared_features)
        
        concated_features = torch.concat([shared_features,private_features], dim=-1)
        exp_logits = self.exp_head(concated_features)
        # exp_logits = exp_logits / math.sqrt(exp_logits.size(-1))

        mask = exp_labels != -100
        if domain_labels is not None:
            loss_domain = self.loss_fn(domain_logits.float().transpose(1,2), domain_labels)
            loss_orthogonality = torch.einsum('LD,LK -> DK', F.normalize(shared_features[mask]), F.normalize(private_features[mask]))
            loss_orthogonality = loss_orthogonality / mask.sum()
            loss_orthogonality = torch.norm(loss_orthogonality)
        else:
            loss_domain = 0
            loss_orthogonality = 0
        
        if exp_labels is not None:
            loss_exp = self.loss_fn(exp_logits.float().transpose(1,2), exp_labels)
        else:
            loss_exp = 0

        loss = loss_exp + loss_domain + loss_orthogonality
        loss = None if loss == 0 else loss

        return {
            'domain_logits': domain_logits,
            'exp_logits': exp_logits,
            'loss_domain': loss_domain,
            'loss_exp': loss_exp,
            'loss_orthogonality': loss_orthogonality,
            'loss':loss
        }
# %%
model = GeneEncoder(num_genes=NUM_GENE, load_g2v=True, vocab_size=VOCAB_SIZE, depth=6, d_model=200, n_heads=10, dropout=0.1, mlp_ratio=4, activation=nn.GELU).to(torch.bfloat16)
model.load_state_dict(torch.load('/mnt/hdd/Shervin/Thesis/prototype1/runs/sanity_check/last.pt'))
# %%
def is_norm_or_bias(n, p):
    return p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower() or "ln" in n.lower()

def build_optimizer(model: nn.Module):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        (no_decay if is_norm_or_bias(n, p) else decay).append(p)
    param_groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)
#%%
optimizer = build_optimizer(model)
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

@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    tot_loss, tot_acc, n_batches = 0.0, 0.0, 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True).long()
        labels    = batch["labels"].to(DEVICE, non_blocking=True).long()
        domain_labels = batch['dataset_labels'].to(DEVICE, non_blocking=True).long()
        # with autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(input_ids, labels, domain_labels)                 # [B, L, V]
        
        loss   = out['loss']

        acc = masked_token_accuracy(out['exp_logits'], labels)
        tot_loss += loss.item()
        tot_acc  += (0.0 if np.isnan(acc) else acc)
        n_batches += 1
    return tot_loss / max(1,n_batches), tot_acc / max(1,n_batches)
#%%
model = model.to(DEVICE)
model.train()

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

step = 0
accum = 0 
optimizer.zero_grad(set_to_none=True)

num_batches=  len(train_loader)
updates_per_epoch = (num_batches + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
total_updates = EPOCHS * updates_per_epoch
warmup_updates = max(1, int(WARMUP_PCT * total_updates))
print(warmup_updates)

min_lr_ratio = FINAL_LR / LR

def cosine_warmup_lambda(step: int):
    """
    step is the number of *completed* optimizer steps seen by the scheduler.
    LambdaLR initializes last_epoch = -1, then after first scheduler.step() it becomes 0.
    """
    if step < warmup_updates:
        # linear warmup from 0 -> 1
        return (step + 1) / warmup_updates
    # cosine from 1 -> min_lr_ratio
    progress = (step - warmup_updates) / max(1, (total_updates - warmup_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine 

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda)
#%%
for epoch in range(EPOCHS):
    for train_batch in train_loader:
        model.train()
        input_ids = train_batch['input_ids'].to(DEVICE)
        labels    = train_batch['labels'].to(DEVICE)
        domain_labels = train_batch['dataset_labels'].to(DEVICE)

        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(input_ids, labels, domain_labels)              # [B, L, 22]

        # with torch.autocast(device_type='cuda', enabled=False):
        # loss_fp32 = loss_fn(logits.float().permute(0, 2, 1), labels)  # [B,22,L] vs [B,L]
        loss_fp32 = out['loss']

        # scale for accumulation to keep LR invariant
        (loss_fp32 / GRAD_ACCUM_STEPS).backward()
        accum += 1

        if accum == GRAD_ACCUM_STEPS:
            total_norm = clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            curr_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{EPOCHS}, step: {step}, loss_exp: {out['loss_exp'].item():.4f}, loss_domain: {out['loss_domain'].item():.4f}, loss_orthogonality: {out['loss_orthogonality'].item():.4f},loss: {loss_fp32.item():.4f}, lr:{curr_lr:.6f}, grad_norm: {total_norm:.4f}')
            tensorboard_writer.add_scalar('loss/train', loss_fp32, step)
            tensorboard_writer.add_scalar('lr', curr_lr, step)
            tensorboard_writer.add_scalar('grad_norm', total_norm, step)

    
    # Handle leftover micro-batches (when len(loader) % GRAD_ACCUM_STEPS != 0)
    if accum > 0:
        total_norm = clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        step += 1

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{EPOCHS}, step: {step}, loss: {loss_fp32.item():.4f}, lr:{curr_lr:.6f}, grad_norm: {total_norm:.4f}')
        tensorboard_writer.add_scalar('loss/train', loss_fp32, step)
        tensorboard_writer.add_scalar('lr', curr_lr, step)
        tensorboard_writer.add_scalar('grad_norm', total_norm, step)
    
    val_loss, val_acc = evaluate(model, test_loader, loss_fn)
    print('#'*10,'Validation','#'*10)
    print(f"[Val] step: {step}, loss:{val_loss:.4f} acc:{val_acc:.4f} ppl={math.exp(val_loss):.2f}")
    tensorboard_writer.add_scalar('loss/val', val_loss, step)
    tensorboard_writer.add_scalar('acc/val', val_acc, step)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'last.pt'))
tensorboard_writer.flush() 
tensorboard_writer.close()