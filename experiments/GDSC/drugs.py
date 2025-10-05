#%%
import pandas as pd
import numpy as np
#%%
drugs_df = pd.read_csv('/home/shervin/Thesis/Code/storage/GDSC/screened_compounds_rel_8.5.csv')
# %%
drugs_df
#%%
gdsc1_drug_response = pd.read_excel('storage/GDSC/drug_screening/GDSC1_fitted_dose_response_27Oct23.xlsx')
gdsc2_drug_response = pd.read_excel('storage/GDSC/drug_screening/GDSC2_fitted_dose_response_27Oct23.xlsx')
# %%
gdsc1_drug_response.head()
# %%
gdsc2_drug_response.head()
# %%
gdsc1_drug_response[gdsc1_drug_response['TCGA_DESC'] == 'BRCA']
#%%
gdsc2_drug_response[gdsc2_drug_response['TCGA_DESC'] == 'BRCA']
#%%
gdsc1_drug_response['DRUG_NAME'].value_counts()
#%%
gdsc2_drug_response['DRUG_NAME'].value_counts()
#%%
import json
with open('temp/clinical_supplement/drugs_data.json') as f:
    tcga_drugs_data = json.load(f)
# %%
tcga_drugs = []
for patient in tcga_drugs_data:
    patient_drugs = tcga_drugs_data[patient]
    for drug in patient_drugs:
        if 'drug_name' in drug:
            tcga_drugs.append(drug['drug_name'])
tcga_drugs = sorted(set(map(lambda x: x.strip().lower(),tcga_drugs)))
# %%
gdsc1_drugs = sorted(set(map(lambda x:x.strip().lower(), gdsc1_drug_response['DRUG_NAME'].tolist())))
gdsc2_drugs = sorted(set(map(lambda x:x.strip().lower(), gdsc2_drug_response['DRUG_NAME'].tolist())))
# %%
set(gdsc1_drugs).intersection(set(tcga_drugs))
#%%
len(set(gdsc1_drugs).intersection(set(gdsc2_drugs)))