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
# %%
with open('temp/meta_data_file_id.json') as f:
    metadata = json.load(f)

sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
#%%
a = sample_sheet[sample_sheet['Data Type'] == 'miRNA Expression Quantification']
# %%
a['sample_type'] = a['Sample ID'].apply(lambda x: x.split('-')[-1][:2])
a['sample_id'] = a['Sample ID'].apply(lambda x: x.split('-')[-1][-1])
a
# %%
stats = a.groupby(by=['Case ID','sample_id','sample_type']).size().unstack(['sample_id', 'sample_type'], fill_value=0)
# %%
stats.to_csv('temp/miRNA-expression/stats.csv')
# %%
