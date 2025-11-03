#%%
import torch
import torch.nn as nn
import time
import torch, torch.nn as nn, torch.nn.functional as F
device = 'cuda'

class FlashSDPA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):  # x: [B,S,D]
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)                     # [B,H,S,Dh]

        # SDPA prefers bf16/fp16 on CUDA; keep outputs in x.dtype
        q = q.to(torch.bfloat16); k = k.to(torch.bfloat16); v = v.to(torch.bfloat16)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]   # [B,1,Sq,Sk] True = mask

        # Prefer flash; allow fallbacks so you don't crash when masks/constraints disable flash
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal
            )
        out = out.transpose(1, 2).contiguous().view(B, S, D).to(x.dtype)
        return self.proj(out)
#%%
# ==== your script ====
device = "cuda"
B, L, D, n_heads = 16, 19221, 200, 10
x = torch.rand((B, L, D), device=device, dtype=torch.bfloat16)

mha = FlashSDPA(D, n_heads).to(device)            # move to GPU
mha = mha.to(dtype=torch.bfloat16)                # <-- make weights bf16
# %%
t1 = time.time()
out = mha(x)
t = time.time() - t1
# %%
a = out.mean()
# %%
a.backward()
# %%
