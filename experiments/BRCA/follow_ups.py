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
follow_up_paths = [
    'temp/clinical_supplement/7_clinical_follow_up_v1.5_brca.csv',
    'temp/clinical_supplement/5_clinical_follow_up_v2.1_brca.csv',
    'temp/clinical_supplement/3_clinical_follow_up_v4.0_brca.csv'
]
# %%
follow_up_dfs = [
    pd.read_csv(path, index_col=0) for path in follow_up_paths
]
#%%
all_patient_ids = []

for df in follow_up_dfs:
    all_patient_ids.extend(df['bcr_patient_barcode'].tolist())
all_patient_ids = list(set(all_patient_ids))
print(len(all_patient_ids))
#%%
not_values = ['[Not Available]', '[Not Applicable]', '[Not Evaluated]']
skip_fields = ['bcr_patient_uuid','bcr_followup_uuid', 'bcr_patient_barcode', 'bcr_followup_barcode']
data = {}

for patient_id in all_patient_ids:
    _dfs = []
    for df in follow_up_dfs:
        _df = df[df['bcr_patient_barcode'] == patient_id].reset_index(drop=True)
        if len(_df) > 0:
            _dfs.append(_df)
    
    patient_follow_up_events = pd.concat(_dfs, join='outer').reset_index(drop=True)
    
    patient_follow_up_events['days_to_last_followup'] = (
        patient_follow_up_events['days_to_last_followup']
        .replace(not_values, np.nan)
    )
    
    patient_follow_up_events['days_to_last_followup'] = pd.to_numeric(
        patient_follow_up_events['days_to_last_followup'], errors='coerce'
    )

    patient_follow_up_events = patient_follow_up_events.sort_values(by='days_to_last_followup', ascending=True)
    patient_follow_up_events = patient_follow_up_events.fillna(not_values[0])

    patient_data = []
    for idx, row in patient_follow_up_events.iterrows():
        _data = {}
        for k,v in row.items():
            if k not in skip_fields and v not in not_values:
                _data[k] = v
        patient_data.append(_data)
    data[patient_id] = patient_data
# %%
count_patient_wise = {}
count_event_wise = {}

for patient_id in tqdm(data):
    for event in data[patient_id]:
        for k,v in event.items():
            if k in count_event_wise:
                if v in count_event_wise[k]:
                    count_event_wise[k][v] += 1
                else:
                    count_event_wise[k][v] = 1
            else:
                count_event_wise[k] = {v:1}
# %%
filtered_count_event_wise = {}
number_of_events = sum(map(lambda x: len(data[x]), data))
for field in count_event_wise:
    num_values = sum([v for k,v in count_event_wise[field].items()])
    if num_values / number_of_events > 0.6:
        filtered_count_event_wise[field] = num_values
# %%
for k in filtered_count_event_wise:
    print(k+':', count_event_wise[k])
#%%
data
#%%
with open('temp/clinical_supplement/follow_ups_data.json', 'w') as f:
    json.dump(data, f)
# %%
