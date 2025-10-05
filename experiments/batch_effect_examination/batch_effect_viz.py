#%%
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from constants import BRCA_DATASET_ROOT_DIR
import os
from tqdm import tqdm
import distinctipy

colors = distinctipy.get_colors(41)
distinctipy.color_swatch(colors)
#%%
sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR,'sample_sheet.tsv'), sep='\t')
sample_sheet
# %%
brca_clinical = pd.read_csv('temp/clinical_supplement/1_clinical_patient_brca.csv', index_col=0)
brca_clinical
# %%
# Loading Gene Expression Data
sample_sheet_gene_expression = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification']

gene_names = None
expressions = None
patient_ids = []
sample_type = []
for idx, row in tqdm(sample_sheet_gene_expression.reset_index(drop=True).iterrows(), total=len(sample_sheet_gene_expression)):
    path = os.path.join(BRCA_DATASET_ROOT_DIR,'data',row['File ID'], row['File Name'])
    _sample_type = row['Sample ID'].split('-')[-1][:-1]
    if _sample_type == '01':
        sample_type.append('Tumor')
    elif _sample_type in ['11','10']:
        sample_type.append('Normal')
    elif _sample_type == '06':
        sample_type.append('Metastisis')
    df = pd.read_csv(path, sep='\t',skiprows=1).iloc[4:]
    if gene_names is None:
        gene_names = df['gene_name'].tolist()
    if expressions is None:
        expressions = np.zeros((len(sample_sheet_gene_expression), len(gene_names)))
    
    expressions[idx, :] = df['tpm_unstranded'].values
    patient_ids.append(row['Case ID'])
# %%
gender = []
vital_status = []
tissue_source_site = []
for patient_id in tqdm(patient_ids):
    row = brca_clinical[brca_clinical['bcr_patient_barcode'] == patient_id]
    if len(row) == 0:
        gender.append(None)
        vital_status.append(None)
        tissue_source_site.append(None)
        print(patient_id)
        continue
    else:
        row = row.iloc[0]

    gender.append(row['gender'])
    vital_status.append(row['vital_status'])
    tissue_source_site.append(row['tissue_source_site'])
#%%
sample_type = np.array(sample_type)
gender = np.array(gender)
vital_status = np.array(vital_status)
tissue_source_site = np.array(tissue_source_site)
#%%
# pca
pca = PCA(n_components=2)
expressions_pca = pca.fit_transform(expressions)
print(pca.explained_variance_ratio_)
#%%
plt.figure(figsize=(15,15))

for axis, labels in enumerate([sample_type, gender, vital_status, tissue_source_site]):
    plt.subplot(2,2,axis+1)
    for idx, value in enumerate(set(labels)):
        indices = labels == value
        plt.scatter(expressions_pca[indices,0], expressions_pca[indices,1], s=15, c=colors[idx], label=value if idx < 5 else None)
    plt.legend()
plt.show()
#%%
tsne = TSNE(n_components=2)
expressions_tsne = tsne.fit_transform(expressions)
#%%
plt.figure(figsize=(15,15))

for axis, labels in enumerate([sample_type, gender, vital_status, tissue_source_site]):
    plt.subplot(2,2,axis+1)
    for idx, value in enumerate(set(labels)):
        indices = labels == value
        plt.scatter(expressions_tsne[indices,0], expressions_tsne[indices,1], s=15, c=colors[idx], label=value if idx < 5 else None)
    plt.legend()
plt.show()
#%%
# TCGA-BH-A0B2 has samples and rna seq and other data modalities but there is no clinical data for this patient !!!!
sample_sheet[sample_sheet['Case ID'] == 'TCGA-BH-A0B2']
#%%
row = sample_sheet[sample_sheet['Case ID'].apply(lambda x: len(x.split(',')) > 2)].iloc[0]
print(os.path.join(BRCA_DATASET_ROOT_DIR, 'data', row['File ID'], row['File Name']))
# %%