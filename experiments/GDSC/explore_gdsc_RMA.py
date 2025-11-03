#%%
import pandas as pd
import numpy as np
from constants import GDSC_DATASET_ROOT_DIR
import os
#%%
gene_expressions = pd.read_csv(os.path.join(GDSC_DATASET_ROOT_DIR, 'gene_expression', 'Cell_line_RMA_proc_basalExp.txt'), sep='\t')
# %%
cell_lines_info = pd.read_excel(os.path.join(GDSC_DATASET_ROOT_DIR, 'Cell_Lines_Details.xlsx'), sheet_name=None)
# %%
cell_lines_info = cell_lines_info['Cell line details']
# %%
a = 'DATA.' + str(int(cell_lines_info.iloc[0]['COSMIC identifier'].item())) 
# %%
gene_expressions['GENE_SYMBOLS'].nunique()
# %%
gene_expressions['GENE_SYMBOLS'].value_counts()
# %%
