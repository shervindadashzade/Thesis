#%%
import time
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
#%%
file_path = '/mnt/windows/GDSC/molecular/expression/rnaseq_all_20250117.csv'
# Count total rows first (skip header)
with open(file_path) as f:
    total_rows = sum(1 for _ in f) - 1  # minus header row
#%%
CHUNK_SIZE= 500_000
total_chunks = total_rows // CHUNK_SIZE + 1
chunked_file = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
# %%
for chunk in chunked_file:
    print(chunk.shape)
    break
# %%
model_names = chunk['model_name'].unique()
with open('experiments/GDSC/model_names.pkl', 'wb') as f:
    pickle.dump(model_names, f)
#%%
with open('experiments/GDSC/model_names.pkl', 'rb') as f:
    model_names = pickle.load(f)
# %%
gene_symbols = None
save_dir = 'temp/GDSC/gene_expression'
for model_name in tqdm(model_names):
    model_gene_expression = None
    chunked_file = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
    for chunk in chunked_file:
        filtered_chunk = chunk[chunk['model_name'] == model_name]
        if model_gene_expression is None:
            model_gene_expression = filtered_chunk
        else:
            model_gene_expression = pd.concat([model_gene_expression, filtered_chunk])
    model_gene_expression.reset_index(drop=True, inplace=True)
    model_gene_symbols = set(model_gene_expression['gene_symbol'].unique())
    if gene_symbols is None:
        gene_symbols = model_gene_symbols
    else:
        gene_symbols = gene_symbols.intersection(model_gene_symbols)
    model_gene_expression = model_gene_expression[model_gene_expression['rsem_tpm'] != 0]
    save_path = os.path.join(save_dir, f'{model_name}.csv')
    model_gene_expression.to_csv(save_path)
#%%
