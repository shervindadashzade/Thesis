import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype1.model.flashsdpa import FlashSDPA

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        hidden = int(mlp_ratio * d_model)

        self.mha = FlashSDPA(d_model, n_heads, dropout=dropout, causal=False)

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.act = activation()
        self.fc2 = nn.Linear(hidden, d_model, bias=True)

        # Dropouts
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Self-attention (q=k=v=x)
        h = self.norm1(x)
        attn_out = self.mha(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_out = self.attn_drop(attn_out)
        x = x + self.resid_drop(attn_out)

        h = self.norm2(x)
        mlp_out = self.fc2(self.mlp_drop(self.act(self.fc1(h))))
        x = x + self.resid_drop(mlp_out)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = FlashSDPA(d_model, n_heads, dropout=dropout, causal=False)

        hidden = int(mlp_ratio * d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.act = activation()
        self.fc2 = nn.Linear(hidden, d_model, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, kv, key_padding_mask=None, attn_mask=None):
        # x attends to kv (q=x, k=v=kv)
        q = self.norm_q(x)
        kvn = self.norm_kv(kv)
        attn_out = self.attn(q, kvn, kvn, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.resid_drop(self.attn_drop(attn_out))

        h = self.norm_ff(x)
        x = x + self.resid_drop(self.fc2(self.mlp_drop(self.act(self.fc1(h)))))
        return x