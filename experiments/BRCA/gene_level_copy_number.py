#%%
import numpy as np
import pandas as pd
import json
import os
from constants import BRCA_DATASET_ROOT_DIR
import matplotlib.pyplot as plt
import re
import shutil
import json
from tqdm import tqdm
# %%
with open('temp/meta_data_file_id.json') as f:
    metadata = json.load(f)

sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
#%%
a = sample_sheet[sample_sheet['Data Type'] == 'Gene Level Copy Number']
#%%
a = sample_sheet[sample_sheet['Data Type'] == 'Gene Level Copy Number']
a = a[a['File Name'].str.contains('ascat3', na=False)]
# %%
b = []
for idx, row in tqdm(a.iterrows(), total=len(a)):
    sample_ids = row['Sample ID'].split(',')
    patient_id = '-'.join(sample_ids[0].split('-')[:-1])
    sample_ids = list(map(lambda x: x.split('-')[-1], sample_ids))
    # sample_type = list(map(lambda x:x[:2], sample_ids))
    # sample_ids = list(map(lambda x:x[-1], sample_ids))
    data = {
        'patient_id':patient_id,  
    }

    for k in sample_ids:
        data[k] = 1
    b.append(data)
# %%
b = pd.DataFrame(b).fillna(0)
# %%
b[b.select_dtypes(include='number').columns] = b[b.select_dtypes(include='number').columns].astype('int8')
# %%
df = b
patient_col = df["patient_id"]
df = df.drop(columns="patient_id")

# Parse column names into (sample_type, vial)
new_cols = [(col[-1], col[:-1]) for col in df.columns]  # (vial, sample_type)

# Assign as MultiIndex
df.columns = pd.MultiIndex.from_tuples(new_cols)

# Put back patient_id
df.insert(0, "patient_id", patient_col)
df = df.sort_values(by='patient_id').reset_index(drop=True)
df.index = df.index + 1
print(df)
# %%
df.to_csv('temp/gene_level_copy_number_variation/stats.csv')
# %%