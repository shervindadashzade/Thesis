#%%
import pandas as pd
import numpy as np
#%%
cell_lines_df = pd.read_excel('storage/GDSC/Cell_Lines_Details.xlsx')
# %%
cell_lines_df.columns = [col.strip().replace('\n',' ') for col in cell_lines_df.columns]
# %%
breast_cancer_filtered = cell_lines_df[cell_lines_df['Cancer Type (matching TCGA label)'] == 'BRCA'].reset_index(drop=True)
# %%
breast_cancer_filtered.to_csv('temp/GDSC/cell_lines.csv')
# %%
mutation_df = pd.read_csv('storage/GDSC/molecular/mutation/mutations_all_20250318.csv')
# %%
mutation_df.head()
# %%
gdsc1 = pd.read_excel('/home/shervin/Thesis/Code/storage/GDSC/drug_screening/GDSC2_fitted_dose_response_27Oct23.xlsx')
# %%
