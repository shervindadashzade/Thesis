#%%
import torch
import pickle
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prototype1.model import FlashSDPA, EncoderBlock, CrossAttentionBlock
from prototype1.dataset import GeneMLMDataset
from prototype1.constants import VOCAB_SIZE, NUM_GENE
#%%
class GeneEncoder(nn.Module):
    def __init__(self, num_genes=NUM_GENE, load_g2v=True, vocab_size = VOCAB_SIZE, depth=3, d_model=200, n_heads=10, dropout=0, mlp_ratio=4.0, activation=nn.GELU):
        super().__init__()
        if load_g2v:
            with open('prototype1/storage/data/gene_embeddings.pkl','rb') as f:
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
        
        out = self.head(out)
        loss = loss_fn(logits.float().transpose(1,2), labels)
        return out
#%%
train_dataset = GeneMLMDataset('prototype1/storage/data/train.pkl')
# %%
device = 'cuda'
model = GeneEncoder().to(device).to(torch.bfloat16)
# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# %%
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
for batch in tqdm(train_loader):
    input_ids = batch['input_ids'].to(device).long()
    labels = batch['labels'].to(device).long()
    logits = model(input_ids)
    loss = loss_fn(logits.float().transpose(1,2), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
# %%
