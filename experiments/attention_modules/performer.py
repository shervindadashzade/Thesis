#%%
import torch
import torch.nn as nn
from performer_pytorch import SelfAttention
import time
# %%
device = 'cuda'
B = 4
L = 16000
D = 200
n_heads = 10
x = torch.rand((B,L,D), device=device)
# %%
mha = SelfAttention(dim=D, heads=n_heads, causal=False).to(device)
# %%
t1 = time.time()
out = mha(x,x,x)
t = time.time() - t1
print(t)
# %%
out.mean().backward()
# %%
