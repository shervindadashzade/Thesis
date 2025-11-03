#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('storage/CCLE/secondary-screen-dose-response-curve-parameters.csv')
# %%
df = df.dropna(subset=['ic50'])
# %%
