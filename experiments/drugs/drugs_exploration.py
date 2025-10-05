#%%
import pandas as pd
import numpy as np
from constants import BRCA_DATASET_ROOT_DIR
import json
# %%
with open('temp/clinical_supplement/drugs_data.json') as f:
    drugs_data = json.load(f)
# %%
drugs = {}
for patient_id in drugs_data:
    for drug in drugs_data[patient_id]:
        if 'drug_name' in drug:
            drug_name = drug['drug_name']
            if drug_name in drugs:
                drugs[drug_name] += 1
            else:
                drugs[drug_name] = 1
# %%
