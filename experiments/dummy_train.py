#%%
from data.datasets import RNASeqDataset
from consts import DATASETS_PATH
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import os.path as osp
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
# %%
LUAD_PATH = osp.join(DATASETS_PATH, 'LUAD')
LUAD_ASSOCIATED_PATH = osp.join(LUAD_PATH, 'RNA-Seq-associated-data')
LUAD_RNA_PATH = osp.join(LUAD_PATH,'RNA-Seq')
sample_sheet = pd.read_csv(osp.join(LUAD_ASSOCIATED_PATH,'sample_sheet.tsv'),sep='\t')
labels = sample_sheet['Sample Type'].apply(lambda x: 0 if x=='Solid Tissue Normal' else 1).values.tolist()
train_sample_sheet, test_sample_sheet = train_test_split(sample_sheet, test_size=0.2, random_state=42, stratify=labels)
train_dataset = RNASeqDataset(sample_sheet=train_sample_sheet, rna_seq_path=LUAD_RNA_PATH)
test_dataset = RNASeqDataset(sample_sheet=test_sample_sheet, rna_seq_path=LUAD_RNA_PATH)



train_labels = train_sample_sheet['Sample Type'].apply(lambda x: 0 if x=='Solid Tissue Normal' else 1).values.tolist()
counter = Counter(train_labels)

weights = [1/counter[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
# %%
train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# %%
# preload train dataset
X = []
Y = []

for x,y in tqdm(train_dataloader):
    X.append(x)
    Y.append(y)
# %%
X = torch.vstack(X)
Y = torch.concat(Y,dim=0)
# %%
train_dataset = TensorDataset(X,Y)
train_dataloader = DataLoader(train_dataset, batch_size=64,shuffle=True)
# %%
model = nn.Sequential(
    nn.LayerNorm(60660),
    nn.Linear(60660, 1),
    nn.Sigmoid()
)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.BCELoss()
epochs = 100
#%%
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for x,y in tqdm(train_dataloader):
        y = y.to(torch.float32).reshape(-1,1)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_dataloader)
    print(f'Epoch [{epoch+1}/{epochs}], loss: {running_loss:.4f}')
    print('Evaluating model...')
    model.eval()
    for x,y in test_dataloader:
        with torch.no_grad():
            y_pred = model(x)
    y_pred = (y_pred.numpy() > 0.5).astype(np.int8).ravel()
    y = y.numpy()
    print(classification_report(y, y_pred))
#%%
    