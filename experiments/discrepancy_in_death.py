#%%
import json
import pandas as pd
import numpy as np
from constants import BRCA_DATASET_ROOT_DIR
import os
from pprint import pprint 
# %%
sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'),sep='\t')
# %%
brca_clinical = pd.read_csv('temp/clinical_supplement/1_clinical_patient_brca.csv', index_col=0)
# %%
brca_clinical
#%%
brca_clinical['vital_status'].value_counts()
# %%
brca_clinical_dead_patient_ids = set(brca_clinical[brca_clinical['vital_status'] == 'Dead']['bcr_patient_barcode'].tolist())
# %%
with open('temp/clinical_supplement/follow_ups_data.json') as f:
    follow_ups = json.load(f)
# %%
follow_ups
#%%
# lets see if there are any patient who reported alive after a dead report?
for patient_id in follow_ups:
    alive_flag = True
    for follow_up in follow_ups[patient_id]:
        if 'vital_status' in follow_up:
            if follow_up['vital_status'] == 'Alive':
                if alive_flag == False:
                    print(patient_id)
                    break
            elif follow_up['vital_status'] == 'Dead':
                continue
#%%
patient_with_more_dead_event = []
for patient_id in follow_ups:
    once_dead_seen = False
    for follow_up in follow_ups[patient_id]:
        if 'vital_status' in follow_up:
            if follow_up['vital_status'] == 'Dead':
                if once_dead_seen:
                    print(patient_id)
                    patient_with_more_dead_event.append(patient_id)
                else:
                    once_dead_seen = True
#%%
patient_with_more_dead_event = list(set(patient_with_more_dead_event))
for patient_id in patient_with_more_dead_event:
    print(patient_id,':')
    pprint(follow_ups[patient_id])
    print('#'*90)
#%%
# lets extract the dead persons recorded in follow ups
follow_ups_dead_patient_ids = []
for patient_id in follow_ups:
    alive_flag = True
    for follow_up in follow_ups[patient_id]:
        if 'vital_status' in follow_up:
            if follow_up['vital_status'] == 'Dead':
                alive_flag = False
                break
    if not alive_flag:
        follow_ups_dead_patient_ids.append(patient_id)
follow_ups_dead_patient_ids = set(follow_ups_dead_patient_ids)
# %%
len(set(follow_ups_dead_patient_ids))
#%%
# %%
len(follow_ups_dead_patient_ids.intersection(brca_clinical_dead_patient_ids))
#%%
len(set(brca_clinical['bcr_patient_barcode'].unique()).intersection(set(follow_ups.keys())))
#%%
a = set(brca_clinical['bcr_patient_barcode'].unique()) - brca_clinical_dead_patient_ids
# %%
len(follow_ups_dead_patient_ids.intersection(brca_clinical_dead_patient_ids))
# %%
len(follow_ups_dead_patient_ids - brca_clinical_dead_patient_ids)
#%%
len(brca_clinical_dead_patient_ids - follow_ups_dead_patient_ids )