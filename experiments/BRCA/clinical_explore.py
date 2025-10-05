#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
#%%
df = pd.read_csv('temp/clinical_supplement/1_clinical_patient_brca.csv', index_col=0)
#%%
df['days_to_death'].value_counts()
#%%
# defining utils
def load_df(path):
    df = pd.read_csv(path, index_col=0)
    not_values = ['[Not Available]', '[Not Applicable]']
    mask = df.apply(lambda col: col.isin(not_values)).mean()
    df = df.loc[:, mask <= 0.5]
    return df
#%%
# clinical patient brca
df = load_df('temp/clinical_supplement/1_clinical_patient_brca.csv')
df
# %%
for col in df:
    print(df[col].value_counts())
    print('#'*100)
#%%
for k,v in df['age_at_initial_pathologic_diagnosis'].value_counts().items():
    print(k,v)
#%%
# NTE
a = pd.read_csv('temp/clinical_supplement/2_clinical_nte_brca.csv', index_col=0)
b = pd.read_csv('temp/clinical_supplement/6_clinical_follow_up_v4.0_nte_brca.csv', index_col=0)
# %%
len(set(a['bcr_patient_barcode'].tolist()).intersection(set(b['bcr_patient_barcode'].tolist())))
# %%
a
# %%
not_values = ['[Not Available]', '[Not Applicable]', '[Not Evaluated]']
#%%
mask = a.apply(lambda col: col.isin(not_values)).mean()
a = a.loc[:, mask <= 0.5]
#%%
mask = b.apply(lambda col: col.isin(not_values)).mean()
b = b.loc[:, mask <= 0.5]
# %%
a.drop(columns=['bcr_patient_uuid'], inplace=True)
b.drop(columns=['bcr_patient_uuid'], inplace=True)
#%%
b.columns = [col if col == 'bcr_patient_barcode' else col+'_follow_up' for col in b.columns]
# %%
c = pd.merge(a,b, on='bcr_patient_barcode', how='outer')
# %%
c.shape
#%%
for col in c.columns:
    print(c[col].value_counts())
    print('#'*100)
# %%
a['bcr_patient_barcode'].value_counts()
#%%
a[a['bcr_patient_barcode'] == 'TCGA-A2-A3XU']
# %%
b['bcr_patient_barcode'].value_counts()
#%%
b[b['bcr_patient_barcode'] == 'TCGA-A2-A04P']
#%%
patient_ids = set(a['bcr_patient_barcode'].tolist() + b['bcr_patient_barcode'].tolist())
#%%
[col if 'day' in col else '' for col in a.columns]
#%%
[col if 'day' in col else '' for col in b.columns]
#%%
patients_nte = {}
for patient_id in patient_ids:
    patients_nte[patient_id] = []
    a_filtered = a[a['bcr_patient_barcode'] == patient_id].sort_values(by='days_to_new_tumor_event_after_initial_treatment')
    b_filtered = b[b['bcr_patient_barcode'] == patient_id].sort_values(by='days_to_new_tumor_event_after_initial_treatment')
    for idx, row in a_filtered.iterrows():
        data = {}
        for k,v in row.items():
            if k == 'bcr_patient_uuid':
                continue
            if v in not_values:
                continue
            data[k] = v
        patients_nte[patient_id].append(data)

    for idx, row in b_filtered.iterrows():
        data = {}
        for k,v in row.items():
            if k == 'bcr_patient_uuid':
                continue
            if v in not_values:
                continue
            data[k] = v
        patients_nte[patient_id].append(data)
# %%
a[a['bcr_patient_barcode'] == 'TCGA-A2-A3XU'].sort_values(by='days_to_new_tumor_event_after_initial_treatment', ascending=True)
#%%
all_patients = sorted(list(set(a['bcr_patient_barcode'].tolist() + b['bcr_patient_barcode'].tolist())))
# %%
c = pd.concat([a,b], join='outer')
c = c.groupby(by='bcr_patient_barcode').apply(lambda x: x.sort_values(by='days_to_new_tumor_event_after_initial_treatment').drop(columns=['bcr_patient_barcode']).reset_index().drop(columns=['index']))
c
#%%
c.head(10)
#%%
not_values = ['[Not Available]', '[Not Applicable]', '[Not Evaluated]']
# %%
from tqdm import tqdm
events = {}

for index in tqdm(c.index):
    patient_id = index[0]
    row = c.loc[index]
    event = {}
    for k,v in row.items():
        if v not in not_values and pd.notna(v):
            event[k] = v
    if patient_id in events:
        events[patient_id].append(event)
    else:
        events[patient_id] = [event]
# %%
with open('temp/clinical_supplement/nte_data.json', 'w') as f:
    json.dump(events, f)
# %%
count_event_wise = {}
count_patient_wise = {}

for patient in tqdm(events):
    keys_seen = []
    for event in events[patient]:
        for k, v in event.items():
            keys_seen.append(k)
            if k in count_event_wise:
                if v in count_event_wise[k]:
                    count_event_wise[k][v] += 1
                else:
                    count_event_wise[k][v] = 1
            else:
                count_event_wise[k] = {v: 1}

    keys_seen = list(set(keys_seen))
    for k in keys_seen:
        if k in count_patient_wise:
            count_patient_wise[k] += 1
        else:
            count_patient_wise[k] = 1
# %%
count_patient_wise
#%%
sum([v for k,v in count_event_wise['days_to_new_tumor_event_after_initial_treatment'].items()])
#%%
drugs_df = load_df('temp/clinical_supplement/9_clinical_drug_brca.csv')
drugs_df
# %%
for col in drugs_df.columns:
    print(drugs_df[col].value_counts())
    print('#'*50)
# %%
drugs_df['bcr_patient_barcode'].value_counts()
# %%
patient_ids = list(set(drugs_df['bcr_patient_barcode'].tolist()))
# %%
not_values = ['[Not Available]', '[Not Applicable]', '[Not Evaluated]']
drugs_data = {}
skip_fields = ['bcr_patient_uuid','bcr_patient_barcode','bcr_drug_barcode','bcr_drug_uuid','form_completion_date','']
for patient_id in patient_ids:
    _df = drugs_df[drugs_df['bcr_patient_barcode'] == patient_id]
    _durgs = []
    for idx, row in _df.iterrows():
        _drug = {}
        for k,v in row.items():
            if k not in skip_fields and v not in not_values:
                _drug[k] = v
        _durgs.append(_drug)
    drugs_data[patient_id] = _durgs
# %%
count_event_wise = {}

for patient_id in tqdm(drugs_data):
    for drug in drugs_data[patient_id]:
        for k,v in drug.items():
            if k in count_event_wise:
                if v in count_event_wise[k]:
                    count_event_wise[k][v] += 1
                else:
                    count_event_wise[k][v] = 1
            else:
                count_event_wise[k] = {v:1}
# %%
count_event_wise
#%%
filtered_count_event_wise = {}
number_of_assoc = sum(map(lambda x: len(drugs_data[x]), drugs_data))
for field in count_event_wise:
    num_values = sum([v for k,v in count_event_wise[field].items()])
    if num_values / number_of_assoc > 0.6:
        filtered_count_event_wise[field] = num_values
# %%
filtered_count_event_wise
# %%
sum([len(drugs_data[patient_id]) for patient_id in drugs_data])
#%%
with open('temp/clinical_supplement/drugs_data.json', 'w') as f:
    json.dump(drugs_data, f)
# %%
