#%%
from constants import BRCA_DATASET_ROOT_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from glob import glob
from pprint import pprint
# %%
sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
# %%
patient_ids = sorted(list(set(sample_sheet['Case ID'].apply(lambda x: [y.strip() for y in x.split(',')]).sum())))
# %%
with open('temp/clinical_supplement/nte_data.json') as f:
    nte_clinical = json.load(f)

with open('temp/clinical_supplement/follow_ups_data.json') as f:
    follow_ups_clinical = json.load(f)

with open('temp/clinical_supplement/drugs_data.json') as f:
    drugs_clinical = json.load(f)

brca_clinical = pd.read_csv('temp/clinical_supplement/1_clinical_patient_brca.csv', index_col=0)
data ={}
for patient_id in tqdm(patient_ids):
    _data = {}
    # checking brca clinical
    num_rows_brca_clinical = brca_clinical[brca_clinical['bcr_patient_barcode'] == patient_id].shape[0]
    _data['brca_clinical # rows'] = num_rows_brca_clinical
        
    num_nte_events = 0
    if patient_id in nte_clinical:
        num_nte_events = len(nte_clinical[patient_id])
    _data['nte_clinical # events'] = num_nte_events

    num_follow_up_events = 0
    if patient_id in follow_ups_clinical:
        num_follow_up_events = len(follow_ups_clinical[patient_id])
    _data['follow_ups_clinical # events'] = num_follow_up_events
    

    num_drugs = 0
    if patient_id in drugs_clinical:
        num_drugs = len(drugs_clinical[patient_id])
    _data['drugs_clinical # drugs'] = num_drugs


    file_counts = sample_sheet[sample_sheet['Case ID'].apply(lambda x: patient_id in x.split(',')[0])]['Data Type'].value_counts()
    

    for modal in ['Methylation Beta Value', 'Protein Expression Quantification', 'Gene Expression Quantification', 'miRNA Expression Quantification', 'Isoform Expression Quantification', 'Gene Level Copy Number', 'Masked Somatic Mutation']:
        field_name = '_'.join(modal.lower().split())+ ' # files'
        if modal not in file_counts:
            count = 0
        else:
            count = file_counts[modal].item()
        _data[field_name] = count

    data[patient_id] = _data
#%%
df = pd.DataFrame.from_dict(data, orient='index')
df
# %%
df.to_csv('temp/3_overall_stats.csv')
# %%