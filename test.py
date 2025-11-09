from bulkrna_bert.datamodule import RNASeqDataModule
from tqdm import tqdm

path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/all_data.pt"

datamodule = RNASeqDataModule()

datamodule.setup("train")

train_loader = datamodule.train_dataloader()


all_dataset_labels = []
for batch in tqdm(train_loader):
    dataset_labels = batch["dataset_labels"]
    all_dataset_labels.append(dataset_labels)
# %%
