tcga_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_input_ids.pt"
ccle_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/ccle_input_ids.pt"
gdsc_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt"
# %%
import torch

tcga_input_ids = torch.load(tcga_path)
ccle_input_ids = torch.load(ccle_path)
gdsc_input_ids = torch.load(gdsc_path)
# %%
tcga_input_ids.shape
ccle_input_ids.shape
gdsc_input_ids.shape
# %%
all_input_ids = torch.concat([ccle_input_ids, gdsc_input_ids, tcga_input_ids], dim=0)
all_input_ids.shape

dataset_labels = (
    [0] * ccle_input_ids.shape[0]
    + [1] * gdsc_input_ids.shape[0]
    + [2] * tcga_input_ids.shape[0]
)
# %%
data = {"input_ids": all_input_ids, "dataset_labels": dataset_labels}

torch.save(data, "bulkrna_bert/data/all_data.pt")
