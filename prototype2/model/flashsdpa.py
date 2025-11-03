import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashSDPA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal 
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):

        B, Sq, D = q.shape
        Bk, Sk, Dk = k.shape
        Bv, Sv, Dv = v.shape
        assert B == Bk == Bv and D == Dk == Dv and Sk == Sv, "Q/K/V shape mismatch"

        H, Dh = self.n_heads, self.head_dim
        assert D == H * Dh, "d_model must be divisible by n_heads"

        q = self.w_q(q).view(B, Sq, H, Dh).transpose(1, 2)  # [B, H, Sq, Dh]
        k = self.w_k(k).view(B, Sk, H, Dh).transpose(1, 2)  # [B, H, Sk, Dh]
        v = self.w_v(v).view(B, Sk, H, Dh).transpose(1, 2)  # [B, H, Sk, Dh]

        out_dtype = q.dtype

        q = q.to(torch.bfloat16); k = k.to(torch.bfloat16); v = v.to(torch.bfloat16)

        if key_padding_mask is not None:
            # Expecting key_padding_mask: [B, Sk], True means "mask out"
            assert key_padding_mask.shape == (B, Sk)
            key_padding_mask = key_padding_mask.to(torch.bool)
            attn_mask = key_padding_mask[:, None, None, :]  # broadcasts over heads and queries

        # Prefer flash; allow fallbacks if masks/constraints disable it
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False if (Sq != Sk) else self.causal  # cross-attn is typically not causal
            )

        out = out.transpose(1, 2).contiguous().view(B, Sq, D).to(out_dtype)
        return self.proj(out)