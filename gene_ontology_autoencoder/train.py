#%%
from gene_ontology_autoencoder.data import GDSCGeneExpressionData
from gene_ontology_autoencoder.models import GOAutoEncoder
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
#%%
with open('storage/go_nn/network_data.pkl','rb') as f:
    network_data = pickle.load(f)
    masks = [network_data['input_mask'], network_data['mask_level_4_to_3'], network_data['mask_level_3_to_2']]
    del network_data

train_indices, test_indices = train_test_split(np.arange(1783), test_size=0.1, shuffle=True, random_state=42)

model = GOAutoEncoder(masks[0],masks[1], masks[2])
train_dataset = GDSCGeneExpressionData(train_indices)
val_dataset = GDSCGeneExpressionData(test_indices, scaler=train_dataset.scaler)
BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)
# %%
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
EPOCHS = 1000
#%%
PLOT_EVERY=50
tsne = TSNE(n_components=2, perplexity=30)
for epoch in range(EPOCHS):
    train_running_loss = 0
    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train()
    for batch_idx, batch in train_progress_bar:
        gene_expressions = batch['gene_expressions']
        labels = batch['labels']

        optimizer.zero_grad()
        x_hat = model(gene_expressions)
        loss = loss_fn(x_hat, gene_expressions)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
        train_progress_bar.set_description(f'Epoch [{epoch+1}/{EPOCHS}], loss:{train_running_loss/(batch_idx+1)}')

    print('evaluating...')
    model.eval()
    val_running_loss = 0
    
    all_val_expressions = []
    all_val_reconstructed = []
    with torch.no_grad():
        for batch in val_loader:
            gene_expressions = batch['gene_expressions']
            labels = batch['labels']
            
            x_hat = model(gene_expressions)
            all_val_expressions.append(gene_expressions)
            all_val_reconstructed.append(x_hat)

            loss = loss_fn(x_hat, gene_expressions)
            val_running_loss += loss.item()
    val_loss = val_running_loss / len(val_loader)
    print(f'val_loss: {val_loss}')
    if (epoch+1) % PLOT_EVERY == 0:
        all_val_expressions = torch.vstack(all_val_expressions).numpy()
        all_val_reconstructed = torch.vstack(all_val_reconstructed).numpy()
        all_data = np.concat([all_val_expressions, all_val_reconstructed], axis=0)
        tsne_embeddings = tsne.fit_transform(all_data)
        colors = ['tab:blue' if i< all_val_expressions.shape[0] else 'tab:red'  for i in range(all_data.shape[0])]
        plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c=colors, s=3, alpha=0.1)
        plt.show()
# %%
