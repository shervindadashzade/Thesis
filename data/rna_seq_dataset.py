#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
import torch.nn.functional as F

class RNATabularDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.expressions = dataset.iloc[:,:-1].values
        self.classes = dataset['label'].unique().tolist()
        self.labels = dataset['label'].apply(lambda x: self.classes.index(x)).values.tolist()

    def __len__(self):
        return self.expressions.shape[0]
        
    def __getitem__(self, index):
        expression = self.expressions[index,:].reshape(1,-1)
        label = self.labels[index]
        
        return expression, label

class RNASeqDataset(Dataset):
    def __init__(self, sample_sheet, rna_seq_path, column='tpm_unstranded'):
        super().__init__()
        self.sample_sheet = sample_sheet
        self.rna_seq_path = rna_seq_path
        self.column = column
    
    def __len__(self):
        return len(self.sample_sheet)

    def __getitem__(self, index):
        sample = self.sample_sheet.iloc[index]
        rna_seq = pd.read_csv(osp.join(self.rna_seq_path,sample['File ID'],sample['File Name']), sep='\t',header=1)
        rna_seq.drop([0,1,2,3],inplace=True)
        expression = rna_seq[self.column].values.astype(np.float32)
        label = sample['Sample Type']
        if label == 'Solid Tissue Normal':
            label = 0
        elif label in ['Primary Tumor','Recurrent Tumor']:
            label=1
        else:
            print(f'Unknown label {label}')
        
        return expression, label

if __name__ == '__main__':
    from consts import DATASETS_PATH
    from tqdm import tqdm
    LUAD_PATH = osp.join(DATASETS_PATH, 'LUAD')
    LUAD_ASSOCIATED_PATH = osp.join(LUAD_PATH, 'RNA-Seq-associated-data')
    LUAD_RNA_PATH = osp.join(LUAD_PATH,'RNA-Seq')

    dataset = RNASeqDataset(sample_sheet_path=osp.join(LUAD_ASSOCIATED_PATH,'sample_sheet.tsv'), rna_seq_path=LUAD_RNA_PATH)
    
    seq, label = dataset[24]
