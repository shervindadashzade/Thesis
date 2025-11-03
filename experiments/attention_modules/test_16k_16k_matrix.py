#%%
import torch
# %%
device = 'cuda'
# %%
L = 16000

a = torch.rand((L,L), dtype=torch.bfloat16, device=device)
# %%
