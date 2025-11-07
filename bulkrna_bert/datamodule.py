import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
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
    ) -> None:
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.mask_percentage = mask_percentage
        self.only_mask = only_mask

    def setup(self, stage: str) -> None:
        if stage == "test":
            self.dataset_test = RNASeqDataset(
                self.path,
                mask_percentage=self.mask_percentage,
                only_mask=self.only_mask,
            )

    def test_dataloader(self):
        return DataLoader(self.dataset_test, self.batch_size, shuffle=False)

    # TODO:: add state handling for training


# %%
if __name__ == "__main__":
    data_module = TCGARNASeqDataModule(mask_percentage=0.12)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        print(input_ids.shape)
        print(labels.shape)
        break
