#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
from constants import BRCA_DATASET_ROOT_DIR
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
# %%
sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
sample_sheet
#%%
gene_level_expression_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification'].reset_index()
del sample_sheet
# %%
# finding number of unique gene_ids without considering the version number
row = gene_level_expression_sample_sheet.iloc[0]
file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'],row['File Name'])
df_file = pd.read_csv(file_path, sep='\t', skiprows=1).loc[4:,]
gene_ids = df_file['gene_id'].tolist()
#%%
data = np.zeros((len(gene_level_expression_sample_sheet), len(gene_ids)))
#%%
patient_ids = []
for idx, row in tqdm(gene_level_expression_sample_sheet.iterrows(), total=len(gene_level_expression_sample_sheet), disable=False):
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'],row['File Name'])
    df_file = pd.read_csv(file_path, sep='\t', skiprows=1).loc[4:,]
    data[idx, :] = df_file['tpm_unstranded'].values
    patient_ids.append(row['Case ID'])
#%%
gene_id_sums = {gene_id:exp_sum for gene_id, exp_sum in zip(gene_ids, data.sum(axis=0).tolist()) }
keep_list = [False] * len(gene_ids)
black_list = []
for idx, gene_id in tqdm(enumerate(gene_id_sums), total=len(gene_id_sums)):
    # print(gene_id)
    if gene_id not in black_list and gene_id_sums[gene_id] != 0:
        base = gene_id.split('.')[0]
        same_base_gene_ids = [(idx,_gene_id, count) for idx, (_gene_id,count) in enumerate(gene_id_sums.items()) if str.startswith(_gene_id, base)]
        if len(same_base_gene_ids) > 1:
            print(same_base_gene_ids)
            max_count = max(same_base_gene_ids, key=lambda x:x[2])
            for item in same_base_gene_ids:
                keep_list[item[0]] = False
                black_list.append(item[1])
            keep_list[max_count[0]] = True
        else:
            keep_list[idx] = True
#%%
keep_list = np.array(keep_list)
gene_ids = np.array(gene_ids)
data = data[:,keep_list]
gene_ids = gene_ids[keep_list]
#%%
_mapping = {item[1]:item[0] for item in df_file[['gene_name','gene_id']].values}
id_to_symbol = {}
#%%
for gene_id in gene_ids:
    pruned_gene_id = gene_id.split('.')[0]
    id_to_symbol[pruned_gene_id] = _mapping[gene_id] 
#%%
save_dir = 'storage/tcga_temp/gene_expression'
tcga_data = {
    'data': data,
    'gene_ids': gene_ids,
    'patient_ids': patient_ids,
    'id_to_symbol': id_to_symbol
}
with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
    pickle.dump(tcga_data, f)
# %%
data.shape
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(data)
#%%
#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne_embeddings = tsne.fit_transform(data)
#%%
map_list = ['01','11','06']
colors = ['red', 'green', 'black']
indices = gene_level_expression_sample_sheet['Sample ID'].apply(lambda x:x.split('-')[-1][:2]).apply(lambda x: map_list.index(x))
c = indices.apply(lambda x: colors[x])
#%%
brca_clinical = pd.read_csv('temp/clinical_supplement/1_clinical_patient_brca.csv',index_col=0)
#%%
brca_clinical[brca_clinical['bcr_patient_barcode'] == 'TCGA-GM-A2DL']['tissue_source_site'].item()
#%%
def get_tissue_source(x):
    barcode = '-'.join(x.split('-')[:-1])
    matched = brca_clinical[brca_clinical['bcr_patient_barcode'] == barcode]['tissue_source_site']
    if len(matched) > 0:
        return matched.iloc[0]  # or .item()
    else:
        return 'GM'
sites = gene_level_expression_sample_sheet['Sample ID'].apply(get_tissue_source)
#%%
sites_unique = sites.unique().tolist()
indices = sites.apply(lambda x: sites_unique.index(x))
colors = [
    'red', 'green', 'blue', 'black', 'orange', 'purple', 'brown', 'pink',
    'gray', 'olive', 'cyan', 'magenta', 'gold', 'lime', 'navy', 'teal',
    'coral', 'orchid', 'crimson', 'indigo', 'darkgreen', 'darkblue',
    'maroon', 'chocolate', 'plum', 'slateblue', 'tomato', 'darkorange',
    'darkcyan', 'seagreen', 'skyblue', 'violet', 'salmon', 'khaki',
    'darkred', 'steelblue', 'rosybrown', 'mediumaquamarine', 'dodgerblue',
    'firebrick', 'peru'
]
#%%
c = indices.apply(lambda x: colors[x])
#%%
#%%
plt.scatter(pca_embeddings[:,0], pca_embeddings[:,1], s=1, c=c)
#%%
plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], s=1, c=c)
plt.legend()
#%%
log_data = np.log2(data + 1)
# %%
vars = log_data.var(axis=0)
#%%
log_data.mean(axis=0)
#%%
plt.plot(np.arange(vars.shape[0]), vars)
#%%
zero_var_indices = np.where(vars == 0)[0]
#%%
expression_example = pd.read_csv(file_path, skiprows=1, sep='\t')
# %%
gene_names = expression_example['gene_name'][4:].values
#%%
gene_names
#%%
gene_names[zero_var_indices]
#%%
expression_example[expression_example['gene_name'] == 'NME1-NME2']
#%%
(data.sum(axis=0) == 0).sum()
#%%
stds = data.std(axis=0)
#%%
stds.min()
# %%
stds.max()
# %%
plt.hist(stds[stds > 1500])