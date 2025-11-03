#%%
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
# %%
with open('prototype2/storage/data/train.pkl', 'rb') as f:
    data = pickle.load(f)
# %%
c = Counter(data['dataset_label'])
# %%
indices = np.arange(data['expressions'].shape[0])
#%%
dataset_labels_np = np.array(data['dataset_label'])
# %%
n_each_class = 20
mask = np.zeros((len(dataset_labels_np),), dtype=np.bool)
for i in range(3):
    indices = np.where(dataset_labels_np == i)[0]
    selected_indices = np.random.choice(indices, n_each_class)
    mask[selected_indices] = True
# %%
val_data = {
    'expressions': data['expressions'][mask,:],
    'model_ids': data['model_ids'][mask],
    'dataset_label': dataset_labels_np[mask].tolist()
}
# %%
train_data = {
    'expressions': data['expressions'][~mask,:],
    'model_ids': data['model_ids'][~mask],
    'dataset_label': dataset_labels_np[~mask].tolist()
}
# %%
with open('prototype2/storage/data/train.pkl','wb') as f:
    pickle.dump(train_data, f)
# %%
with open('prototype2/storage/data/val.pkl','wb') as f:
    pickle.dump(val_data, f)
# %%
