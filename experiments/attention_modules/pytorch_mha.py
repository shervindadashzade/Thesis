#%%
import torch
import torch.nn as nn
import time
# %%
device = 'cuda'
B = 1
L = 16000
D = 200
n_heads = 10
x = torch.rand((B,L,D), device=device, dtype=torch.bfloat16)
q = torch.rand((B,1000, D), device=device, dtype=torch.bfloat16)
# %%
mha = nn.MultiheadAttention(D, n_heads, batch_first=True, device=device).to(torch.bfloat16)
# %%
t1 = time.time()
with torch.no_grad():
    out,_ = mha(x,x,x)
t = time.time() - t1
# %%
