#%%
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class GDSCGeneExpressionData(Dataset):
    def __init__(self, indices=None, scaler=None):
        super().__init__()
        with open('/home/shervin/Thesis/Code/gene_ontology_autoencoder/storage/gdsc_data.pkl','rb') as f:
            data = pickle.load(f)
            self.data = data['gene_expression']
            self.labels = np.array(data['tcga_labels'])
            del data

        if indices is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data)
        else:
            self.scaler = scaler

        # self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)

        self.classes = sorted(list(set(self.labels.tolist())))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        gene_expression = self.data[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]
        return {
            'gene_expressions': gene_expression,
            'labels': label_idx
        }

    def __len__(self):
        return self.data.shape[0]

# %%
if __name__ == '__main__':
    dataset = GDSCGeneExpressionData()
    print(dataset[0])