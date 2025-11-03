#%%
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# %%
base_dir = 'storage/processed'

with open(os.path.join(base_dir, 'gdsc.pkl'),'rb') as f:
    gdsc_data = pickle.load(f)

with open(os.path.join(base_dir, 'ccle.pkl'),'rb') as f:
    ccle_data = pickle.load(f)

with open(os.path.join(base_dir, 'tcga.pkl'),'rb') as f:
    tcga_data = pickle.load(f)
# %%
gene_names = gdsc_data['gene_names']
tcga_expression = tcga_data['expression'].T
gdsc_expression = gdsc_data['expression'].T
ccle_expression = ccle_data['expression'].T
# %%
ccle_indices = np.arange(0, len(ccle_expression))
gdsc_indices = np.arange(0, len(gdsc_expression))
tcga_indices = np.arange(0, len(tcga_expression))
# %%
ccle_train_indices, ccle_test_indices = train_test_split(ccle_indices, test_size=0.1, random_state=42)
gdsc_train_indices, gdsc_test_indices = train_test_split(gdsc_indices, test_size=0.1, random_state=42)
tcga_train_indices, tcga_test_indices = train_test_split(tcga_indices, test_size=0.015, random_state=42)
# %%
tcga_model_ids = np.array(tcga_data['model_ids'])
gdsc_model_ids = np.array(gdsc_data['model_ids'])
ccle_model_ids = np.array(ccle_data['model_ids'])
# %%
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

n_bins = 20
encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
# %%
train_expressions = np.concat([ccle_expression[ccle_train_indices], gdsc_expression[gdsc_train_indices], tcga_expression[tcga_train_indices]], axis=0)
train_dataset_label = [0] * len(ccle_train_indices) + [1] * len(gdsc_train_indices) + [2] * len(tcga_train_indices)
train_model_ids = np.concat([ccle_model_ids[ccle_train_indices], gdsc_model_ids[gdsc_train_indices], tcga_model_ids[tcga_train_indices]], axis=0)
# %%
test_expressions = np.concat([ccle_expression[ccle_test_indices], gdsc_expression[gdsc_test_indices], tcga_expression[tcga_test_indices]], axis=0)
test_dataset_label = [0] * len(ccle_test_indices) + [1] * len(gdsc_test_indices) + [2] * len(tcga_test_indices)
test_model_ids = np.concat([ccle_model_ids[ccle_test_indices], gdsc_model_ids[gdsc_test_indices], tcga_model_ids[tcga_test_indices]], axis=0)
# %%
train_expressions = encoder.fit_transform(train_expressions)
test_expressions = encoder.transform(test_expressions)
# %%
train_data = {
    'expressions': train_expressions,
    'model_ids': train_model_ids,
    'dataset_label': train_dataset_label
}
test_data = {
    'expressions': test_expressions,
    'model_ids': test_model_ids,
    'dataset_label': test_dataset_label
}
# %%
with open('prototype1/storage/data/train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('prototype1/storage/data/test.pkl', 'wb') as f:
    pickle.dump(test_data, f)