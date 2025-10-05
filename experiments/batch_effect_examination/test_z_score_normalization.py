#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
# %%
with open('storage/gene_expressions_cleaned/tcga.pkl','rb') as f:
    tcga_data = pickle.load(f)
with open('storage/gene_expressions_cleaned/gdsc.pkl','rb') as f:
    gdsc_data = pickle.load(f)
# %%
tcga_log_transformed = np.log1p(tcga_data['data'])
gdsc_log_transformed = np.log1p(gdsc_data['data'])
#%%
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
tcga_normalized = standard_scaler.fit_transform(tcga_log_transformed)
gdsc_normalized = standard_scaler.fit_transform(gdsc_log_transformed)
# %%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
all_data = np.concat([tcga_normalized, gdsc_normalized])
tsne_embeddings = tsne.fit_transform(all_data)
colors = ['tab:blue' if i < tcga_normalized.shape[0] else 'tab:orange' for i in range(all_data.shape[0])]
plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], s=3, c= colors, alpha=0.3)
# %%
