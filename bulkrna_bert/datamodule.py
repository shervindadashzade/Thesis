import lightning as L
from typing import Any
from torch.utils.data import random_split
import math
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import os
from bulkrna_bert.utils import mask_input_ids


class RNASeqDataset(Dataset):
    def __init__(
        self,
        path: str = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/input_ids.pt",
        mask_percentage: float = 0.15,
        only_mask=True,
    ) -> None:
        super().__init__()
        self.path = path
        self.inputs = torch.load(path)
        self.mask_percentage = mask_percentage
        self.only_mask = only_mask

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index) -> dict:
        input = self.inputs[index]
        data = mask_input_ids(
            input, mask_percentage=self.mask_percentage, only_mask=self.only_mask
        )
        return data


class TCGARNASeqDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/input_ids.pt",
        batch_size: int = 16,
        mask_percentage=0.15,
        only_mask=True,
        train_portion=0.8,
        val_portion=0.1,
        test_portion=0.1,
    ) -> None:
        super().__init__()
        self.current_train_batch_index = 0
        self.path = path
        self.batch_size = batch_size
        self.mask_percentage = mask_percentage
        self.only_mask = only_mask
        self.train_portion = train_portion
        self.val_portion = val_portion
        self.test_portion = test_portion
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        dataset = RNASeqDataset(self.path, self.mask_percentage, self.only_mask)
        generator = torch.Generator().manual_seed(42)
        dataset_size = len(dataset)

        train_size = math.ceil(dataset_size * self.train_portion)
        val_size = math.ceil(dataset_size * self.val_portion)
        test_size = dataset_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, self.batch_size, shuffle=False, num_workers=55
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=False, num_workers=55
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=False, num_workers=55
        )

    def state_dict(self):
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]

    # TODO:: add state handling for training


# %%
if __name__ == "__main__":
    data_module = TCGARNASeqDataModule(
        path="/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt",
        mask_percentage=0.12,
        only_mask=False,
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        print(input_ids.shape)
        print(labels.shape)
        break
