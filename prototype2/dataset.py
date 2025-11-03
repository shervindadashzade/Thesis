import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from prototype1.constants import VOCAB_SIZE
from prototype1.utils import mlm_mask_torch

class GeneMLMDataset(Dataset):
    def __init__(self, dataset_file, mask_prob=0.15, do_mask=True):
        super().__init__()
        self.mask_prob = mask_prob
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)

        self.expressions = data['expressions']
        self.model_ids = data['model_ids']
        self.dataset_labels = data['dataset_label']
        self.do_mask = do_mask
    
    def __getitem__(self, idx):
        input_ids = self.expressions[idx] + 2
        input_ids = np.concat([np.array([0]), input_ids], axis=0).astype(np.int8)
        input_ids = torch.from_numpy(input_ids)
        model_id = self.model_ids[idx].item()
        if self.do_mask:
            input_ids, labels, mask_idx = mlm_mask_torch(input_ids, VOCAB_SIZE, mask_prob=self.mask_prob)
            labels = labels.long()
            dataset_label = (torch.zeros_like(labels) + -100).long()
            dataset_label[mask_idx] = self.dataset_labels[idx]
        else:
            dataset_label = self.dataset_labels[idx]

        if self.do_mask:
            return {
                'input_ids': input_ids.long(),
                'labels': labels,
                'mask_ids': mask_idx,
                'model_ids': model_id,
                'dataset_labels': dataset_label
            }
        else:
            return {
                'input_ids': input_ids.long(),
                'model_ids': model_id,
                'dataset_labels': dataset_label
            }
    def __len__(self):
        return len(self.expressions)

if __name__ == '__main__':
    dataset = GeneMLMDataset('prototype1/storage/data/test.pkl')