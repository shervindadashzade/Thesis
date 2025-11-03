#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv('storage/GDSC/rna_seq_ebi/E-MTAB-3983-query-results.tpmss.tsv',skiprows=4,sep='\t')
# %%
df2 = pd.read_excel('storage/GDSC/Cell_Lines_Details.xlsx', sheet_name=None)
# %%
sheet = df2['Cell line details']
# %%
sheet
# %%
cell_line_names = list(map(lambda x: x.split(',')[0], df.columns[2:].tolist()))
# %%
a = sheet['Sample Name'].tolist()
# %%
len(set(cell_line_names).intersection(set(a)))
# %%
passport_rna_seq = pd.read_csv('storage/GDSC/rna_seq_passports/rnaseq_merged_rsem_tpm_20250922.csv')
# %%
passport_rna_seq
#%%
model_names = passport_rna_seq.iloc[0].tolist()[3:]
# %%
len(a)
# %%
len(model_names)
#%%
len(set(a).intersection(set(model_names)))
# %%
len(model_names)
# %%
len(set(model_names))
# %%
ccle_expressions = pd.read_csv('storage/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv')
# %%
ccle_expressions
# %%
