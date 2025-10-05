#%%
import pandas as pd
from constants import GDSC_DATASET_ROOT_DIR
import matplotlib.pyplot as plt
import os
# %%
# Cell Lines Details.xlsx
path = os.path.join(GDSC_DATASET_ROOT_DIR, 'Cell_Lines_Details.xlsx')
df = pd.read_excel(path, sheet_name=None)

print(f'Different sheets names:')
for sheet_name in df.keys():
    print(sheet_name)
# %%
# start by Cell line details
sheet = df['Cell line details'].iloc[:-1]
sheet.columns = [col_name.replace('\n','').strip() for col_name in sheet.columns]
sheet.head()
# %%
# extracting statistics
stats = {}
counting_sheet = sheet.iloc[:, 2:6]
for col in counting_sheet:
    counts = counting_sheet[col].value_counts()
    stats[col] = counts['Y'].item()
    print(counts)
    print('#'*100)
# %%
plt.figure(figsize=(15,5))
bars = plt.bar(stats.keys(), stats.values())
plt.title('Cell Lines Information Distribution')
plt.xlabel('Omics')
plt.ylabel('Count')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,  # x-position
        height,                           # y-position
        f'{height:.0f}',                  # text (integer)
        ha='center', va='bottom'          # horizontal/vertical alignment
    )

plt.show()
#%%
sheet = df['COSMIC tissue classification']
sheet.head()
#%%
sheet['Line'].nunique()
#%%
for col in ['Site','Histology']:
    count = sheet[col].value_counts()
    print(count)
#%%