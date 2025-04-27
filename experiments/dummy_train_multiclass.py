#%%
from data.datasets import RNATabularDataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as osp
from collections import Counter
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
# %%
expression_data = pd.read_csv('storage/epxression_tpm.csv')
#%%
train_data, test_data = train_test_split(expression_data, test_size=0.1, shuffle=True, random_state=10, stratify=expression_data['label'].values)
#%%
train_dataset = RNATabularDataset(train_data)
test_dataset = RNATabularDataset(test_data)
#del expression_data
del train_data
del test_data
# %%
counter = Counter(train_dataset.labels)
# %%
weights = [1/counter[label] for label in train_dataset.labels]
# %%
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset))
# %%
train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
#%%
class FeatureMaskNetwork(nn.Module):
    def __init__(self, input_features_dim, steps=3, hidden_dim=128):
        super().__init__()
        self.steps = steps
        self.linear_layers = [nn.Linear(hidden_dim, hidden_dim) for i in range(steps)]
        self.layer_norms = [nn.LayerNorm(hidden_dim) for i in range(steps)]

        self.downsample_network = nn.Linear(input_features_dim,hidden_dim)
        self.upsample_network = nn.Linear(hidden_dim, input_features_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.downsample_network(x)

        for step in range(self.steps):
            out = self.linear_layers[step](x)
            x = self.layer_norms[step](x + out)
        
        x = self.upsample_network(x)
        
        out = self.activation(x)
        
        return out

class Network(nn.Module):
    def __init__(self,input_features_dim=60660, steps=3):
        super().__init__()
        
        self.feature_mask_network = FeatureMaskNetwork(input_features_dim, steps)

        self.classification_head = nn.Sequential(
        nn.LayerNorm(60660),
        nn.Linear(60660, 512),
        nn.LayerNorm(512),
        nn.Linear(512,128),
        nn.LayerNorm(128),
        nn.Sigmoid(),
        nn.Linear(128,3),
        nn.Softmax(dim=1)
        )

    def forward(self,x):
        mask = self.feature_mask_network(x)

        class_prob = self.classification_head(x * mask)

        return class_prob        


model = Network()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
epochs = 1000
#%%
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    y_preds = []
    y_actual = []
    for x,y in tqdm(train_dataloader):
        x = x.to(torch.float32).squeeze()
        y = y.to(torch.long)
        optimizer.zero_grad()
        y_pred = model(x)

        y_preds.append(y_pred)
        y_actual.append(y)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_dataloader)
    losses.append(running_loss)
    
    y_preds = torch.argmax(torch.concat(y_preds,dim=0),dim=1)
    y_actual = torch.concat(y_actual, dim=0)
    train_accuracy = accuracy_score(y_actual, y_preds)

    print(f'Epoch [{epoch+1}/{epochs}], loss: {running_loss:.4f}, accuracy: {train_accuracy:.4f}')
    print('Evaluating model...')
    model.eval()
    for x,y in test_dataloader:
        x = x.to(torch.float32).squeeze()
        with torch.no_grad():
            y_pred = model(x)
    y_pred = torch.argmax(y_pred,dim=1).to(torch.long)
    y = y.numpy()
    print(classification_report(y, y_pred, target_names=train_dataset.classes))
#%%
with torch.no_grad():
    mask = model.feature_mask_network(x)
# %%
mean_mask = torch.abs(mask.mean(dim=0))
# %%
indices = torch.where(mean_mask> 0.99)[0]
# %%
expression_data.columns[:-1][indices]
# %%
df_sample = pd.read_csv('/home-old/shervin/datasets/LUAD/RNA-Seq/fe20855f-d3ce-4e02-b2dd-3df15eab44bf/564e41d6-f9a4-4a6b-bb97-f8e5423b84bd.rna_seq.augmented_star_gene_counts.tsv',sep='\t',header=1)
# %%
df_sample[df_sample['gene_id'].isin(expression_data.columns[:-1][indices])]['gene_name'].values
# %%

# %%
