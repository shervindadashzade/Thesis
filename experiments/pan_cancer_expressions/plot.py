#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.manifold import TSNE
from constants import PANCANCER_DATASET_ROOT_DIR
from sklearn.model_selection import train_test_split
# %%
with open(os.path.join(PANCANCER_DATASET_ROOT_DIR,'data.pkl'), 'rb') as f:
    tcga_data = pickle.load(f)
# %%
tcga_cancer_types = tcga_data['cancer_types']
tcga_gene_ids = tcga_data['gene_ids']
tcga_data = tcga_data['gene_expressions']
#%%
tcga_data = np.log1p(tcga_data)
# %%
with open('gene_ontology_autoencoder/storage/gdsc_data.pkl', 'rb') as f:
    gdsc_data = pickle.load(f)
# %%
gdsc_cancer_types = gdsc_data['tcga_labels']
gdsc_gene_ids = gdsc_data['gene_ids']
gdsc_data = gdsc_data['gene_expression']
#%%
gdsc_data = np.log1p(gdsc_data)
#%%
tcga_gene_ids = tcga_gene_ids.tolist()
#%%
tcga_gene_bases = [gene_id.split('.')[0] for gene_id in tcga_gene_ids]
#%%
tcga_selected_gene_ids = [tcga_gene_bases.index(gene_id) for gene_id in gdsc_gene_ids]
#%%
tcga_data = tcga_data[:, tcga_selected_gene_ids]
#%%
#‌ sampling cause the system can not handle this
_, test_ge, _, test_labels = train_test_split(tcga_data, cancer_types, test_size=0.1, shuffle=True, random_state=42, stratify=cancer_types)
#%%
from collections import Counter
c = Counter(test_labels)
print(c)
# %%
tsne_embeddings = TSNE().fit_transform(test_ge)
# %%
test_labels = np.array(test_labels)
# %%
unique_labels = np.unique(test_labels)
# %%
plt.figure(figsize=(8, 6))
fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})

unique_labels = np.unique(test_labels)

# --- main scatter plot ---
for label in unique_labels:
    indices = np.where(test_labels == label)[0]
    _embeddings = tsne_embeddings[indices]
    ax.scatter(_embeddings[:, 0], _embeddings[:, 1], s=3, label=label)

ax.set_title("t-SNE Embeddings")
ax_legend.axis('off')  # hide axes for legend subplot

# --- separate legend subplot ---
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc='center', frameon=False)

plt.tight_layout()
plt.show()
# %%
# %%
