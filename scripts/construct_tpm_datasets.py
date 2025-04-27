#%%
import pandas as pd
import numpy as np
from consts import DATASETS_PATH
import os.path as osp
import os
from tqdm import tqdm
from npy_append_array import NpyAppendArray
#%%
# first constructing LUAD TPM expressions dataset
LUAD_BASE_PATH = osp.join(DATASETS_PATH,'LUAD')
LUAD_ASSOCIATED_PATH = osp.join(LUAD_BASE_PATH,'RNA-Seq-associated-data')
LUAD_RNA_PATH = osp.join(LUAD_BASE_PATH, 'RNA-Seq')

luad_sample_sheet = pd.read_csv(osp.join(LUAD_ASSOCIATED_PATH,'sample_sheet.tsv'),sep='\t')
# %%
labels= []
gene_ids = None
expression_file = osp.join('storage','tpm_expression.npy')
if os.path.exists(expression_file):
    os.remove(expression_file)

for idx, row in tqdm(luad_sample_sheet.iterrows(), total=len(luad_sample_sheet)):
    expression = pd.read_csv(osp.join(LUAD_RNA_PATH,row['File ID'],row['File Name']), sep='\t',header=1).drop([0,1,2,3])
    
    expression_levels = expression['tpm_unstranded'].values.reshape(1,-1)
    
    if gene_ids is None:
        gene_ids = expression['gene_id'].values
    
    if row['Sample Type'] in ['Primary Tumor','Recurrent Tumor']:
        label = 'LUAD'
    else:
        label = 'Control'
    
    with NpyAppendArray(expression_file) as npaa:
        npaa.append(expression_levels)
    labels.append(label)
#%%

# %%
# constructing the LUSC TPM expressions dataset
LUSC_BASE_PATH = osp.join(DATASETS_PATH,'LUSC')
LUSC_ASSOCIATED_PATH = osp.join(LUSC_BASE_PATH,'RNA-Seq-associated-data')
LUSC_RNA_PATH = osp.join(LUSC_BASE_PATH, 'RNA-Seq')

lusc_sample_sheet = pd.read_csv(osp.join(LUSC_ASSOCIATED_PATH,'sample_sheet.tsv'),sep='\t')
# %%
for idx, row in tqdm(lusc_sample_sheet.iterrows(), total=len(lusc_sample_sheet)):
    expression = pd.read_csv(osp.join(LUSC_RNA_PATH,row['File ID'],row['File Name']), sep='\t',header=0).drop([0,1,2,3])
    
    expression_levels = expression['tpm_unstranded'].values.reshape(1,-1)
    
    if row['Sample Type'] in ['Primary Tumor','Recurrent Tumor']:
        label = 'LUSC'
    else:
        label = 'Control'
    
    with NpyAppendArray(expression_file) as npaa:
        npaa.append(expression_levels)
    labels.append(label)
# %%
data = np.load(expression_file)
# %%
expression_data = pd.DataFrame(data, columns=gene_ids)
expression_data['label'] = labels
# %%
expression_data.to_csv(osp.join('storage','epxression_tpm.csv'),index=False)
# %%