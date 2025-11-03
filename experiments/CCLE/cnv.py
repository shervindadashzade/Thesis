#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('storage/CCLE/OmicsCNGeneWGS.csv', index_col=0)
# %%
df
# %%
a = df.iloc[:,6:].values
# %%
a[a==np.nan] = 0
# %%
a = np.nan_to_num(a, nan=0)
# %%
a.max()
# %%
a.min()
# %%
df2 = pd.read_csv('storage/CCLE/PortalOmicsCNGeneLog2.csv', index_col=0)
# %%
a = df2.values
# %%
a.min()
#%%
a.max()
# %%
df = pd.read_csv('storage/GDSC/cnv/WGS_purple_genes_cn_category_20250207.csv')
# %%
df = pd.read_csv('storage/CCLE/OmicsSomaticMutationsMatrixDamaging.csv', index_col=0)
# %%
df
# %%
a = list(map(lambda x: x.split()[0], df.columns[5:].tolist()))
# %%
import pickle
# %%
with open('storage/processed/gene_names.pkl', 'rb') as f:
    gene_names = pickle.load(f)
# %%
len(a)
#%%
len(set(a))
# %%
