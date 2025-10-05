#%%
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# %%
with open('storage/go_nn/network_data.pkl', 'rb') as f:
    network_data = pickle.load(f)
# %%
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.register_buffer('mask', torch.tensor(mask))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)
# %%
input_mask = network_data['input_mask']
mask_level_4_to_3 = network_data['mask_level_4_to_3']
mask_level_3_to_2 = network_data['mask_level_3_to_2']
input_features = network_data['input_gene_ids']

class GOAutoEncoder(nn.Module):
    def __init__(self, input_mask, mask_4to3, mask_3to2):
        super().__init__()
        self.encoder = nn.Sequential(
            MaskedLinear(input_mask.shape[1], input_mask.shape[0], input_mask),
            nn.GELU(),
            MaskedLinear(mask_4to3.shape[1], mask_4to3.shape[0], mask_4to3),
            nn.GELU(),
            MaskedLinear(mask_3to2.shape[1], mask_3to2.shape[0], mask_3to2),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            MaskedLinear(mask_3to2.shape[0], mask_3to2.shape[1], mask_3to2.T),
            nn.GELU(),
            MaskedLinear(mask_4to3.shape[0], mask_4to3.shape[1], mask_4to3.T),
            nn.GELU(),
            MaskedLinear(input_mask.shape[0], input_mask.shape[1], input_mask.T)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model = GOAutoEncoder(input_mask, mask_level_4_to_3, mask_level_3_to_2)
del network_data
# %%
num_params = sum([p.numel() for p in model.parameters()])
trainable_params = 0
for m in model.modules():
    if isinstance(m, MaskedLinear):
        trainable_params += torch.sum(m.mask).item()
print(f'all parameters: {num_params}, trainable_parameters: {trainable_params}, density: {trainable_params/num_params}')
# %%
with open('storage/gdsc_temp/gene_expression/data.pkl','rb') as f:
    gdsc_data = pickle.load(f)
# %%
gene_indices = np.array([idx for idx, gene_id in enumerate(gdsc_data['gene_ids']) if gene_id in input_features])
expression_data = np.log1p(gdsc_data['data'][:,gene_indices])

del gdsc_data
# %%
expression_data = torch.tensor(expression_data, dtype=torch.float32)
#%%
del model
model = GOAutoEncoder(input_mask, mask_level_4_to_3, mask_level_3_to_2)
optimizer = torch.optim.AdamW(model.parameters())
loss_fn = nn.MSELoss()
EPOCHS = 100
#%%
EPOCHS = EPOCHS * 2
model.eval()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(expression_data)
    loss = loss_fn(out, expression_data)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{EPOCHS}], loss: {loss.item()}')
#%%
tsne = TSNE(n_components=2, perplexity=30)
model.eval()
with torch.no_grad():
    out = model.encoder(expression_data)
tsne_embeddings = tsne.fit_transform(out.numpy())
plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1])
# %%
