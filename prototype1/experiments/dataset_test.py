#%%
import pickle

with open('prototype1/storage/data/train.pkl', 'rb') as f:
    data = pickle.load(f)
# %%
expressions = data['expressions']
# %%
data = expressions[0]
# %%
data += 2
# %%
import numpy as np
# %%
mask_idx = 1
np.concat([[0], data], axis=0)
# %%
len(np.unique(expressions + 2))
# %%
indices_to_alter = np.random.choice(np.arange(0, len(data)), size=int(0.15*len(data)))
# %%
def mlm_mask_numpy(
    input_ids: np.ndarray,
    vocab_size: int,
    mask_prob: float = 0.15,
    mask_token_id: int = 1,   # MSK
    special_token_ids: set = {0, 1},  # CLS=0, MSK=1
    rng: np.random.Generator | None = None,
):
    assert input_ids.ndim == 1, "expects a 1D sequence"
    rng = rng or np.random.default_rng()

    seq = input_ids.copy()
    labels = np.full_like(seq, fill_value=-100)

    # candidate positions = not special tokens
    candidates = np.where(~np.isin(seq, list(special_token_ids)))[0]
    n_to_mask = max(1, int(round(mask_prob * candidates.size))) if candidates.size else 0
    if n_to_mask == 0:
        return seq, labels, np.array([], dtype=int)

    mask_idx = rng.choice(candidates, size=n_to_mask, replace=False)
    labels[mask_idx] = seq[mask_idx]

    n80 = int(round(0.8 * n_to_mask))
    n10 = int(round(0.1 * n_to_mask))
    n10_keep = n_to_mask - n80 - n10   # remainder to avoid off-by-one

    idx80 = mask_idx[:n80]
    idx10_rand = mask_idx[n80:n80+n10]
    idx10_keep = mask_idx[n80+n10:]

    # 80% → MSK
    seq[idx80] = mask_token_id

    # 10% → random token (not special, not original)
    low = 2  # since 0,1 are reserved; you already offset real tokens by +2
    if vocab_size <= low:
        raise ValueError("vocab_size must be > 2 to sample random tokens.")
    rand_tokens = rng.integers(low, vocab_size, size=idx10_rand.size, dtype=seq.dtype)
    # avoid collisions with originals
    collisions = rand_tokens == input_ids[idx10_rand]
    while np.any(collisions):
        rand_tokens[collisions] = rng.integers(low, vocab_size, size=np.sum(collisions), dtype=seq.dtype)
        collisions = rand_tokens == input_ids[idx10_rand]
    seq[idx10_rand] = rand_tokens

    # 10% → keep (no change)
    # seq[idx10_keep] stays as is

    return seq, labels, mask_idx
# %%
data = np.concat([np.array([0]), data], axis=0)
#%%
# %%
vocab_size = len(np.unique(expressions)) + 2
#%%
seq, labels, mask_idx = mlm_mask_numpy(data.astype(int), vocab_size=vocab_size)
#%%
import torch
from torch.utils.data import Dataset
# %%
with open('storage/processed/gene_names.pkl', 'rb') as f:
    gene_names = pickle.load(f)
# %%
g2vec = {}
with open('storage/gene2vec/gene2vec_dim_200_iter_9_w2v.txt','r') as f:
    g2vec_lines = f.readlines()[1:]
# %%
from tqdm import tqdm
for line in tqdm(g2vec_lines):
    parts = line.split()
    gene_name = parts[0]
    embeddings = np.array([float(p) for p in parts[1:]])
    g2vec[gene_name] = embeddings
# %%
gene_embeddings = np.zeros((len(gene_names), 200))
for idx, gene_name in tqdm(enumerate(gene_names), total=len(gene_names)):
    gene_embeddings[idx] = g2vec[gene_name] 
# %%
sim = gene_embeddings @ gene_embeddings.T
# %%
for i in range(len(sim)):
    sim[i,i] = 0
# %%
with open('prototype1/storage/data/gene_embeddings.pkl','wb') as f:
    pickle.dump(gene_embeddings, f)
# %%
