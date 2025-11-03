#%%
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
# %%
# loading gene names
with open('storage/processed/gene_names.pkl','rb') as f:
    gene_names = pickle.load(f)
#%%
# extract CCLE expressions
df = pd.read_csv('storage/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv', index_col=0)
df
# %%
ccle_model_ids = df['ModelID'].tolist()
# %%
df.columns = list(map(lambda x: x.split()[0] if len(x.split())>1 else x, df.columns))
# %%
df
# %%
ccle_expressions = np.zeros((len(gene_names), len(ccle_model_ids)), dtype=np.float32)
# %%
for idx, gene_name in tqdm(enumerate(gene_names), total=len(gene_names)):
    ccle_expressions[idx] = df[gene_name].values
# %%
ccle_expressions.shape
# %%
ccle_data = {
    'gene_names': gene_names,
    'model_ids': ccle_model_ids,
    'expression': ccle_expressions
}
# %%
with open('storage/processed/ccle.pkl', 'wb') as f:
    pickle.dump(ccle_data,f)
# %%
df = pd.read_csv('storage/GDSC/rna_seq_passports/rnaseq_merged_rsem_tpm_20250922.csv')
# %%
gdsc_model_names = df.iloc[0].tolist()[3:]
# %%
df = df.iloc[3:]
# %%
gdsc_expressions = np.zeros((len(gene_names), len(gdsc_model_names)), dtype=np.float32)
# %%
for idx, gene_name in tqdm(enumerate(gene_names),total=len(gene_names)):
    gdsc_expressions[idx] = df[df['model_id'] == gene_name].values[0,3:]
# %%
np.nan_to_num(gdsc_expressions, copy=False, nan=0.0)
# %%
gdsc_data = {
    'gene_names': gene_names,
    'model_ids': gdsc_model_names,
    'expression': gdsc_expressions
}
# %%
with open('storage/processed/gdsc.pkl','wb') as f:
    pickle.dump(gdsc_data,f)
# %%
tcga_sample_sheet = pd.read_csv('storage/TCGA/PAN/sample_sheet.tsv',sep='\t')
#%%
sample_ids = tcga_sample_sheet['Sample ID'].tolist()
# %%
tcga_expressions = np.zeros((len(gene_names), len(sample_ids)), dtype=np.float32)
#%%
tcga_base_dir='storage/TCGA/PAN/gene_expressions/data'
for sample_idx, row in tqdm(tcga_sample_sheet.iterrows(), total=len(tcga_sample_sheet)):

    file_path = os.path.join(tcga_base_dir, row['File ID'], row['File Name'])

    df = pd.read_csv(file_path, skiprows=1, sep='\t')

    a = df[df['gene_name'].isin(gene_names)][['gene_name', 'tpm_unstranded']].groupby(by='gene_name').max().reindex(gene_names)

    tcga_expressions[:, sample_idx] = a.to_numpy().flatten()
#%%
tcga_expressions = np.log1p(tcga_expressions)
#%%
tcga_data = {
    'gene_names': gene_names,
    'model_ids': sample_ids,
    'expression': tcga_expressions
}

with open('storage/processed/tcga.pkl','wb') as f:
    pickle.dump(tcga_data, f)
# %%