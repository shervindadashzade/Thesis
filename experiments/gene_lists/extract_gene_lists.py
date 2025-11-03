#%%
import pandas as pd
import numpy as np
import os
import pickle
# %%
df = pd.read_csv('storage/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv', index_col=0)
# %%
df
# %%
ccle_gene_names = df.columns[5:].tolist()
# %%
df = pd.read_csv('storage/GDSC/rna_seq_passports/rnaseq_merged_rsem_tpm_20250922.csv')
#%%
# %%
gdsc_gene_names = df['model_id'].tolist()[3:]
# %%
a = list(map(lambda x: x.split()[0] ,ccle_gene_names))
# %%
b = gdsc_gene_names
# %%
c = set(a).intersection(set(b))
len(c)
#%%
df = pd.read_csv('storage/TCGA/PAN/gene_expressions/data/00a1a02a-2b45-4065-81c0-dd886efe8464/9a69296a-e334-4387-8a3e-43201d647d2d.rna_seq.augmented_star_gene_counts.tsv', sep='\t', skiprows=1)
df = df[df['tpm_unstranded']!= 0]
# %%
df
# %%
tcga_gene_names = df['gene_name'].tolist()[4:]
# %%
d = list(set(tcga_gene_names).intersection(c))
#%%
tcga_gene_counts = df['gene_name'].value_counts()
# %%
for gene_name in d:
    count = tcga_gene_counts[gene_name].item()
    if count > 1:
        print(gene_name, count)
# %%
with open('storage/gene2vec/gene2vec_dim_200_iter_9_w2v.txt') as f:
    g2v_lines = f.readlines()
# %%
g2v_lines = g2v_lines[1:]
# %%
g2v_gene_names = list(map(lambda x: x.split()[0], g2v_lines))
# %%
gene_names = list(set(g2v_gene_names).intersection(d))
# %%
len(gene_names)
# %%
type(gene_names)
# %%
import pickle
# %%
with open('storage/processed/gene_names.pkl','wb') as f:
    pickle.dump(gene_names, f)
# %%
