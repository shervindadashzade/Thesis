#%%
import numpy as np
import pandas as pd
import json
import os
from constants import BRCA_DATASET_ROOT_DIR
import matplotlib.pyplot as plt
import re
import shutil
# %%
with open(os.path.join(BRCA_DATASET_ROOT_DIR, 'metadata.json')) as f:
    metadata = json.load(f)
#%%
metadata_file_id = {}
for data in metadata:
    file_id = data['file_id']
    metadata_file_id[file_id] = data
with open('temp/meta_data_file_id.json', 'w') as f:
    json.dump(metadata_file_id, f)
#%%
with open('temp/meta_data_file_id.json') as f:
    metadata = json.load(f)
#%%
sample_sheet = pd.read_csv(os.path.join(BRCA_DATASET_ROOT_DIR, 'sample_sheet.tsv'), sep='\t')
# %%
sample_sheet.groupby(by=['Data Category','Data Type']).size().plot(kind='barh')
#%%
sample_sheet.groupby(by=['Data Category','Data Type']).size()
# %%
case_id_list = sorted(list(set([case_id.strip() for case_ids in sample_sheet['Case ID'].apply(lambda x: x.split(',')) for case_id in case_ids])))
# %%
rows_without_sample_id = sample_sheet[sample_sheet['Sample ID'].apply(lambda x: isinstance(x, float))]
# %%
rows_without_sample_id['Data Category'].unique()
#%%
rows_without_sample_id['Data Type'].unique()
# %%
sample_ids = sorted(list(set([sample_id.strip() for sample_id_list in sample_sheet['Sample ID'].apply( lambda x: x.split(',') if not isinstance(x, float) else None).dropna() for sample_id in sample_id_list])))
sample_ids.remove('')
# %%
dictionary_types = {
    '01': 'Tumor',
    '10': 'Normal',
    '11': 'Solid Tissue Normal',
    '06': 'Metastatic'
}
sample_types = [dictionary_types[sample_id.split('-')[-1][:2]] for sample_id in sample_ids]
# %%
samples_df = pd.DataFrame({'sample_id':sample_ids, 'sample_type':sample_types})
#%%
samples_df.groupby('sample_type').size().plot(kind='bar')
samples_df.groupby('sample_type').size()
#%%
samples_df['case_id'] = samples_df['sample_id'].apply(lambda x: '-'.join(x.split('-')[:3]))
#%%
samples_df['sample_portion'] = samples_df['sample_id'].apply(lambda x: x.split('-')[-1][2:])
# %%
samples_df['sample_id'].apply(lambda x: x.split('-')[-1][:2]).unique()
#%%
samples_df[['case_id', 'sample_type', 'sample_portion']].groupby(by=['sample_type','sample_portion']).size()
#%%
a = samples_df[['case_id', 'sample_portion','sample_type']].groupby(by=['case_id','sample_portion']).apply(lambda x:x['sample_type'])
#%%
df_data = {}
for idx, type_sample in a.items():
    case_id = idx[0]
    if case_id not in df_data:
        df_data[case_id] = { idx[1] : [type_sample]}
    else:
        if idx[1] in df_data[case_id]:
            df_data[case_id][idx[1]].append(type_sample)
        else:
            df_data[case_id][idx[1]] = [type_sample]
# %%
a = pd.DataFrame.from_dict(df_data, orient='index').fillna(0)
#%%
a.to_csv('temp/1.csv')
#%%
df_data = {}
for idx, row in sample_sheet.iterrows():
    row_case_id_list = row['Case ID'].split(',')
    row_case_id_list = list(map(lambda x:x.strip(), row_case_id_list))
    if len(row_case_id_list) < 3:
        row_case_id_list = [row_case_id_list[0]]
    for case_id in row_case_id_list:
        if case_id not in df_data:
            df_data[case_id] = {
                row['Data Type'] : 1
            }
        else:
            if row['Data Type'] in df_data[case_id]:
                df_data[case_id][row['Data Type']] += 1
            else:
                df_data[case_id][row['Data Type']] = 1
# %%
a = pd.DataFrame.from_dict(df_data, orient='index').fillna(0).astype(int)
# %%
a.to_csv('temp/2_patients_files.csv')
# %%
for modal in a.columns:
    ax = a[modal].value_counts().plot(kind='bar')
    plt.title(modal)

    # Add labels on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.05,  # x, y
                int(height), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
# %%
a[a['Pathology Report'] > 1]['Pathology Report']
#%%
pathology_report_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Pathology Report']
pathology_report_sample_sheet
# %%
pathology_report_sample_sheet.to_csv('temp/pathology_reports.csv')
# %%
a_ = pathology_report_sample_sheet[pathology_report_sample_sheet['Case ID'] == 'TCGA-BH-A1ES']
# %%
for idx, row in a_.iterrows():
    print(row['Tumor Descriptor'])
    print(os.path.join(BRCA_DATASET_ROOT_DIR,'data',row['File ID']))
# %%
clinical_supplement_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Clinical Supplement']
clinical_supplement_sample_sheet
# %%
a[a['Clinical Supplement'] == 11]['Clinical Supplement']
#%%
def check_case_id(case_id, case_id_value):
    case_ids = [x.strip() for x in case_id_value.split(',')]
    return case_id in case_ids
#%%
test_case_id = 'TCGA-E2-A158'
a_ = clinical_supplement_sample_sheet[clinical_supplement_sample_sheet['Case ID'].apply(lambda x: check_case_id(test_case_id, x))]
# %%
for idx, row in a_.iloc[2:].reset_index().iterrows():
    save_name = f'{idx+1}_'+'_'.join('.'.join(row['File Name'].split('.')[1:-1]).split('_')[1:]) + '.csv'
    save_path = os.path.join('temp/clinical_supplement', save_name)
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
    df = pd.read_csv(file_path, sep='\t', skiprows=1).drop(0)
    df.to_csv(save_path)
    # print(file_path)
    print(save_name)
# %%
case_id = 'TCGA-BH-A0B2'
clinical_supplement_sample_sheet[clinical_supplement_sample_sheet['Case ID'].apply(lambda x: check_case_id(case_id, x))]
# %%
a_ = pd.read_csv('temp/clinical_supplement/9_clinical_drug_brca.csv', index_col=0)
#%%
not_values = ['[Not Available]', '[Not Applicable]']
#%%
mask = a_.apply(lambda col: col.isin(not_values)).mean()
# %%
a_ = a_.loc[:, mask <= 0.5]
# %%
a_.apply(lambda col: col.isin(not_values)).mean()
# %%
for col in a_.columns:
    counts = a_[col].value_counts()
    # if len(counts) < 0.5 * len(a_) and col != 'form_completion_date':
        # print(counts)
    print(counts)
    print('#'*100)
# %%
a_1 = pd.read_csv('temp/clinical_supplement/7_clinical_follow_up_v1.5_brca.csv', index_col=0)
a_2 = pd.read_csv('temp/clinical_supplement/5_clinical_follow_up_v2.1_brca.csv', index_col=0)
a_3 = pd.read_csv('temp/clinical_supplement/3_clinical_follow_up_v4.0_brca.csv', index_col=0)
# %%
mask = a_1.apply(lambda col: col.isin(not_values)).mean()
a_1 = a_1.loc[:, mask <= 0.5]

mask = a_2.apply(lambda col: col.isin(not_values)).mean()
a_2 = a_2.loc[:, mask <= 0.5]
#%%
a_1_patients = set(a_1['bcr_patient_barcode'].tolist())
a_2_patients = set(a_2['bcr_patient_barcode'].tolist())
# %%
common_a1_a2 = list(a_1_patients.intersection(a_2_patients))
#%%
a_1[a_1['bcr_patient_barcode'] == common_a1_a2[0]]
# %%
a_2[a_2['bcr_patient_barcode'] == common_a1_a2[0]]
# %%
a_3[a_3['bcr_patient_barcode'] == common_a1_a2[0]]
#%%
a[a['Masked Intensities'] == 6]['Masked Intensities']
# %%
sample_sheet_masked_intensities = sample_sheet[sample_sheet['Data Type'] == 'Masked Intensities']
# %%
a_ = sample_sheet_masked_intensities[sample_sheet_masked_intensities['Case ID'] == 'TCGA-E2-A15K']
# %%
for idx, row in a_.iterrows():
    # file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data','')
    print(row['Sample ID'],row['File Name'])
#%%
a[a['Methylation Beta Value'] == 3]['Methylation Beta Value']
#%%
sample_sheet_methylation_beta_valus = sample_sheet[sample_sheet['Data Type'] == 'Methylation Beta Value']
# %%
a_ = sample_sheet_methylation_beta_valus[sample_sheet_methylation_beta_valus['Case ID'] == 'TCGA-E2-A15K']
# %%
for idx, row in a_.iterrows():
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
    print(row['Sample ID'])
    print(file_path)
    a__ = pd.read_csv(file_path, sep='\t', header=None, names=['probe','beta_value'])
    break
a__
#%%
a__.to_csv('temp/dna_methylation_beta_value/TCGA-E2-A15K.csv')
# %%
a__['probe'].apply(lambda x: re.sub(r'\d+', '', x)).unique()
# %%
a[a['Gene Level Copy Number'] == 6]['Gene Level Copy Number']
#%%
sample_sheet_gene_level_copy_number = sample_sheet[sample_sheet['Data Type'] == 'Gene Level Copy Number']
# %%
a_ = sample_sheet_gene_level_copy_number[sample_sheet_gene_level_copy_number['Case ID'].apply(lambda x: x.split(',')[0] == 'TCGA-A7-A26E')]
# %%
for idx, row in a_.iterrows():
    print(row['Sample ID'],'\t', row['File Name'])
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data', row['File ID'], row['File Name'])
    dest_dir = f'temp/gene_level_copy_number_variation/{row['File Name']}'
    shutil.copy(file_path, dest_dir)
# %%
b = sample_sheet_gene_level_copy_number['File Name'].apply(
    lambda x: '-'.join(x.split('-')[-1].split('.')[1:3])
).value_counts()

plt.figure(figsize=(12, 5)) 
ax = b.plot(kind='barh')
plt.title('Gene Level Copy Number Variations')

for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 0.05,  
            p.get_y() + p.get_height() / 2.,
            int(width),
            ha='left', va='center')

plt.tight_layout()
plt.show()
# %%
a[a['Gene Level Copy Number'] == 1]['Gene Level Copy Number']
#%%
patients_with_one_cnv_file = a[a['Gene Level Copy Number'] == 1]['Gene Level Copy Number'].index.tolist()
#%%
one_files_cnv = sample_sheet_gene_level_copy_number[sample_sheet_gene_level_copy_number['Case ID'].apply(lambda x: x.split(',')[0].strip() in patients_with_one_cnv_file)]
# %%
one_files_cnv
#%%
b = one_files_cnv['File Name'].apply(
    lambda x: '-'.join(x.split('-')[-1].split('.')[1:3])
).value_counts()
plt.figure(figsize=(12, 5)) 
ax = b.plot(kind='barh')
plt.title('Gene Level Copy Number Variations')

for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 0.05,  
            p.get_y() + p.get_height() / 2.,
            int(width),
            ha='left', va='center')

plt.tight_layout()
plt.show()
#%%
sample_sheet_gene_level_copy_number[sample_sheet_gene_level_copy_number['File Name'].apply(lambda x: 'ascat3' in x)]['Sample ID']
#%%
patients_with_one_protein = a[a['Protein Expression Quantification'] == 1]['Protein Expression Quantification'].index.tolist()
#%%
sample_sheet_protein_expression = sample_sheet[sample_sheet['Data Type']=='Protein Expression Quantification']
#%%
a_ = sample_sheet_protein_expression[sample_sheet_protein_expression['Case ID'].apply(lambda x: x in patients_with_one_protein)]
#%%
a_['Sample ID'].apply(lambda x: x.split('-')[-1]).value_counts() 
#%%
a[a['Protein Expression Quantification'] == 3]['Protein Expression Quantification']
# %%
# %%
a_ = sample_sheet_protein_expression[sample_sheet_protein_expression['Case ID'] == 'TCGA-BH-A1FE']
#%%
for idx, row in a_.iterrows():
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data', row['File ID'], row['File Name'])
    dest_path = os.path.join('temp/protein_expression_quantification',row['Sample ID']+'.tsv')
    print(row['File Name'])
    # shutil.copy(file_path, dest_path)
# %%
a[a['Gene Expression Quantification'] == 4]['Gene Expression Quantification']
#%%
sample_sheet_gene_expression = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification']
# %%
a_ = sample_sheet_gene_expression[sample_sheet_gene_expression['Case ID'] == 'TCGA-A7-A0DB']
#%%
for idx, row in a_.iterrows():
    print(row['Sample ID'],'\t', row['File Name'])
#%%
a__ = []
for idx, row in a_.reset_index().iloc[[1,2]].iterrows():
    # print(row['File Name'])
    for data in metadata:
        if data['file_name'] == row['File Name']:
            break
    print(row['Sample ID'], row['File Name'])
    # print(data['analysis']['updated_datetime'])
    path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
    a__.append(pd.read_csv(path, sep='\t', skiprows=1))
    for key in data:
        print(key,':',data[key])
    print('#'*100)
#%%
a__[0]
#%%
a__[1]
#%%
for idx, row in a_.reset_index().iloc[[1,2]].iterrows():
    file_path = os.path.join(BRCA_DATASET_ROOT_DIR,'data',row['File ID'], row['File Name'])
    for data in metadata:
        if data['file_name'] == row['File Name']:
            break
    dest = os.path.join('temp/gene_expression_quantification',data['associated_entities'][0]['entity_submitter_id']+'_' + row['File Name'])
    shutil.copy(file_path, dest)
    
# %%
a__ = a[a['Gene Expression Quantification'] == 1]['Gene Expression Quantification'].index.to_list()
#%%
sample_sheet_gene_expression[sample_sheet_gene_expression['Case ID'].isin(a__)]['Sample ID'].apply(lambda x: x.split('-')[-1]).value_counts()
# %%
a[a['miRNA Expression Quantification'] == 4]['miRNA Expression Quantification']
#%%
a[a['miRNA Expression Quantification'] == 4]['miRNA Expression Quantification']
#%%
sample_sheet_mirna = sample_sheet[sample_sheet['Data Type'] == 'miRNA Expression Quantification']
# %%
a_ = sample_sheet_mirna[sample_sheet_mirna['Case ID'] == 'TCGA-A7-A0DB']
# %%
for idx, row in a_.iterrows():
    if '01A' in row['Sample ID']:
        file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
        _meta_data = metadata[row['File ID']]
        full_sample_id = _meta_data['associated_entities'][0]['entity_submitter_id']
        dest_path = os.path.join('temp/miRNA-expression',f'{full_sample_id}.tsv')
        shutil.copy(file_path, dest_path)
# %%
b = pd.read_csv('/home/shervin/Thesis/Code/temp/miRNA-expression/TCGA-A7-A0DB-01A-11R-A010-13.tsv',sep='\t')
b
# %%
b_ = pd.read_csv('/home/shervin/Thesis/Code/temp/gene_expression_quantification/TCGA-A7-A0DB-01A-11R-A00Z-07_b4aad532-f92b-43d9-820c-f3c98e382225.rna_seq.augmented_star_gene_counts.tsv', sep='\t', skiprows=1)
b_
#%%
b
#%%
b__ = b_[b_['gene_type']== 'miRNA']
#%%
b__[b__!=0]
#%%
a[a['Isoform Expression Quantification'] == 4]['Isoform Expression Quantification']
#%%
sample_sheet_isoform_expression = sample_sheet[sample_sheet['Data Type'] == 'Isoform Expression Quantification']
# %%
a_ = sample_sheet_isoform_expression[sample_sheet_isoform_expression['Case ID'] == 'TCGA-A7-A0DB']
# %%
for idx, row in a_.iterrows():
    if '01A' in row['Sample ID']:
        file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
        print(file_path)
        _meta_data = metadata[row['File ID']]
        full_sample_id = _meta_data['associated_entities'][0]['entity_submitter_id']
        dest_path = os.path.join('temp/isoform_expression',f'{full_sample_id}.tsv')
        shutil.copy(file_path, dest_path)
# %%
a[a['Masked Somatic Mutation'] == 3]['Masked Somatic Mutation']
#%%
sample_sheet_masked_somatic_mutation = sample_sheet[sample_sheet['Data Type'] == 'Masked Somatic Mutation']
# %%
a_ = sample_sheet_masked_somatic_mutation[sample_sheet_masked_somatic_mutation['Case ID'].apply(lambda x: x.split(',')[0].strip() == 'TCGA-A7-A0DB')]
# %%
for idx, row in a_.iterrows():
    if '01A' in row['Sample ID']:
        file_path = os.path.join(BRCA_DATASET_ROOT_DIR, 'data',row['File ID'], row['File Name'])
        _meta_data = metadata[row['File ID']]
        full_sample_id = _meta_data['associated_entities'][0]['entity_submitter_id']
        full_sample_id_1 = _meta_data['associated_entities'][1]['entity_submitter_id']
        # print(full_sample_id, file_path)
        dest_path = os.path.join('temp/masked_somatic_mutation',f'{full_sample_id}-{full_sample_id_1}.{row['File Name'].split('.')[-1]}')
        # print(dest_path)
        shutil.copy(file_path, dest_path)
# %%
a__ = pd.read_csv('temp/masked_somatic_mutation/TCGA-A7-A0DB-01A-11D-A272-09-TCGA-A7-A0DB-10A-02D-A272-09/d577f725-e910-4a77-89e9-2e982d5493e5.wxs.aliquot_ensemble_masked.maf', sep='\t',comment='#')
# %%
for k,v in a__.iloc[0].items():
    print(k,'\t',v)
# %%
columns = []
for col in a__.columns:
    if a__[col].isna().sum() / len(a__) < 0.5:
        columns.append(col)
# %%
a__[columns].to_csv('temp/masked_somatic_mutation/explore.csv')
# %%
