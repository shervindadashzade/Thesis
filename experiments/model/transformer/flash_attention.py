#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
#%%
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
# %%
x = torch.rand(4,16000, 200).to('cuda').to(torch.bfloat16)
q = torch.rand(4,1000, 200).to('cuda').to(torch.bfloat16)
mha = FlashSDPA(200, 10, 0, False).to('cuda').to(torch.bfloat16)

out_self = mha(x,x,x)
print(out_self.shape)
out_cross = mha(q,x,x)
print(out_cross.shape)
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
#%%
class GeneEncoder(nn.Module):
    def __init__(self,num_genes, num_exp_bin=10, num_cnv=3, num_mut=2, depth=3, d_model=200, n_heads=10, dropout=0, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        self.g2v_embedding = nn.Embedding(num_genes, d_model)
        self.exp_embedding = nn.Embedding(num_exp_bin, d_model)
        self.cnv_embedding = nn.Embedding(num_cnv, d_model)
        self.mut_embedding = nn.Embedding(num_mut, d_model)
        
        self.depth = depth
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout, mlp_ratio, activation) for i in range(depth)])
    
    def forward(self, gene_ids, exps, cnvs, muts):
        gene_embeddings = self.g2v_embedding(gene_ids)
        exp_embeddings = self.exp_embedding(exps)
        cnvs_embeddings = self.cnv_embedding(cnvs)
        mut_embeddings = self.mut_embedding(muts)

        x = gene_embeddings + exp_embeddings + cnvs_embeddings + mut_embeddings
        out = x
        for layer in self.layers:
            out = layer(out)
        
        return out
# %%
num_genes = 16000
num_exp_bin = 20
num_mut = 2
num_cnv = 3
device = 'cuda'
gene_ids = torch.arange(0,num_genes, dtype=torch.long).reshape(1,-1).to(device)
exps = torch.randint(0, num_exp_bin, (1, num_genes), dtype=torch.long).to(device)
cnvs = torch.randint(0, num_cnv, (1, num_genes), dtype=torch.long).to(device)
muts = torch.randint(0, num_mut, (1, num_genes), dtype=torch.long).to(device)
# %%
model = GeneEncoder(num_genes, num_exp_bin, num_cnv, num_mut, depth=3).to(device).to(torch.bfloat16)
# %%
out = model(gene_ids, exps, cnvs, muts)
print(out.shape)
# %%
a = nn.MultiheadAttention(200, 10, batch_first=True).to(device).to(torch.bfloat16)
# %%
a(out,out,out)
# %%
sum([p.numel() for p in model.parameters()])
# %%
torch.save(model.state_dict(),'test.pth')