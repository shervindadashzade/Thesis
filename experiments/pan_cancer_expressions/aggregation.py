#%%
from constants import PANCANCER_DATASET_ROOT_DIR
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
# %%
sample_sheet = pd.read_csv(os.path.join(PANCANCER_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
# %%
sample_sheet['tumor_classification'] = sample_sheet['Sample ID'].apply(lambda x: x.split('-')[-1][:2])
print(sample_sheet['tumor_classification'].value_counts())
#%%
sample_sheet_filtered = sample_sheet[sample_sheet['tumor_classification'] == '01'].reset_index()
# %%
row = sample_sheet_filtered.iloc[0]
path = os.path.join(PANCANCER_DATASET_ROOT_DIR, 'data', row['File ID'], row['File Name'])
gene_expression = pd.read_csv(path, sep='\t', skiprows=1).iloc[4:]
gene_ids = gene_expression['gene_id'].unique()
gene_id_bases = gene_expression['gene_id'].apply(lambda x: x.split('.')[0]).unique()
#%%
gene_expressions = np.zeros((sample_sheet_filtered.shape[0], len(gene_ids)), dtype=np.float32)
#%%
for idx, row in tqdm(sample_sheet_filtered.iterrows(), total=len(sample_sheet_filtered)):
    path = os.path.join(PANCANCER_DATASET_ROOT_DIR, 'data', row['File ID'], row['File Name'])
    gene_expression = pd.read_csv(path, sep='\t', skiprows=1).iloc[4:]
    gene_expressions[idx] = gene_expression['tpm_unstranded'].values
# %%
gene_indices_zero = np.where(gene_expressions.sum(axis=0) == 0)[0]
gene_ids = np.delete(gene_ids, gene_indices_zero, axis=0)
gene_expressions = np.delete(gene_expressions, gene_indices_zero, axis=1)
# %%
gene_bases = np.array([gene_id.split('.')[0] for gene_id in gene_ids])
gene_bases_unique = np.unique(gene_bases)
# %%
cancer_types = sample_sheet_filtered['Project ID'].values
# %%
data = {
    'gene_ids': gene_ids,
    'gene_expressions': gene_expressions,
    'cancer_types': cancer_types
}
# %%
with open(os.path.join(PANCANCER_DATASET_ROOT_DIR, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)
# %%
