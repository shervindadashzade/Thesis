#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from consts import DATASETS_PATH
import os.path as osp
import os
#%%
luad_base_path = osp.join(DATASETS_PATH,'LUAD')
luad_rna_seq_path = osp.join(luad_base_path, 'RNA-Seq')
luad_associated_data_path = osp.join(luad_base_path, 'RNA-Seq-associated-data')
# %%
# what is sample_sheet
sample_sheet = pd.read_csv(osp.join(luad_associated_data_path,'sample_sheet.tsv'),sep='\t')
# %%
sample_sheet.iloc[0]
# %%
sample_sheet['Case ID'].value_counts()
# %%
sample_sheet['Sample Type'].value_counts()
# %%
# what is pathology detail.tsv
pathology_detail = pd.read_csv(osp.join(luad_associated_data_path, 'pathology_detail.tsv'), sep='\t')
# %%
pathology_detail
# %%
follow_up = pd.read_csv(osp.join(luad_associated_data_path, 'follow_up.tsv'), sep='\t')
# %%
follow_up
# %%
family_history = pd.read_csv(osp.join(luad_associated_data_path, 'family_history.tsv'), sep='\t')
# %%
family_history
# %%
exposure = pd.read_csv(osp.join(luad_associated_data_path, 'exposure.tsv'), sep='\t')
# %%
exposure
# %%
clinical = pd.read_csv(osp.join(luad_associated_data_path, 'clinical.tsv'), sep='\t')
# %%
clinical
# %%
clinical.columns
# %%
for col in clinical.columns:
    if col in ['case_id','case_submitter_id','project_id', 'age_at_index']:
        continue
    counts =clinical[col].value_counts() 
    if len(counts) > 1:
        print(counts)
        print('#'*20)
# %%
# exploring RNA-Seq data

def sample_sequence(sample_sheet: pd.DataFrame):
    sample = sample_sheet.sample(1).iloc[0]
    rna_seq = pd.read_csv(osp.join(luad_rna_seq_path,sample['File ID'],sample['File Name']),sep='\t', header=1)
    rna_seq.drop([0,1,2,3],axis=0, inplace=True)
    return rna_seq

sample1 = sample_sequence(sample_sheet)
sample2 = sample_sequence(sample_sheet)
sample3 = sample_sequence(sample_sheet)
#%%