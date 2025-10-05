#%%
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
# %%
with open('storage/gdsc_temp/gene_expression/data.pkl', 'rb') as f:
    gdsc_data = pickle.load(f)
    gdsc_gene_expression = gdsc_data['data']
    gdsc_model_names = gdsc_data['model_names']
    gdsc_gene_ids = gdsc_data['gene_ids']
    del gdsc_data
# %%
with open('storage/go_nn/network_data.pkl', 'rb') as f:
    network_data = pickle.load(f)
    input_gene_ids = network_data['input_gene_ids']
    del network_data
# %%
gene_indices = [gdsc_gene_ids.index(gene_id) for gene_id in input_gene_ids]
expression_data = gdsc_gene_expression[:, gene_indices]
# %%
cell_line_details = pd.read_excel('/mnt/windows/GDSC/Cell_Lines_Details.xlsx')
# %%
cell_line_details.columns = [col.strip().replace('\n','') for col in cell_line_details.columns]
cell_line_details.columns
# %%
cell_line_details = cell_line_details[['Sample Name', 'Cancer Type(matching TCGA label)']]
#%%
cell_line_details.columns = ['sample_name','tcga_label']
cell_line_details = cell_line_details.set_index('sample_name').dropna()
# %%
tcga_label = []
for model_name in gdsc_model_names:
    if model_name in cell_line_details.index:
        label = cell_line_details.loc[model_name]['tcga_label']
        print(label)
        if label == 'UNABLE TO CLASSIFY':
            label='UNK'
    else:
        label='UNK'
    tcga_label.append(label)
# %%
data = {
    'gene_expression': np.log1p(expression_data),
    'gene_ids': input_gene_ids,
    'model_names': gdsc_model_names,
    'tcga_labels': tcga_label
}

with open('gene_ontology_autoencoder/storage/gdsc_data.pkl','wb') as f:
    pickle.dump(data,f)
# %%
