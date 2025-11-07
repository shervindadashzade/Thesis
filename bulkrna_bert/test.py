from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import torch
from bulkrna_bert.model import BulkRNABert

repo = "InstaDeepAI/BulkRNABert"
device = "cuda:0"
model = BulkRNABert.from_pretrained(repo)
model.eval()
# model.to(torch.bfloat16)
model.to(device)
input_ids = torch.load("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/input_ids.pt")
input_ids = input_ids[0].unsqueeze(0)

input_ids.shape
input_ids[0, 0] = 0
input_ids = input_ids.to(device)

out = model(input_ids)
out["logits"].shape
# %%
# tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
# model = AutoModel.from_pretrained(repo, trust_remote_code=True)
# %%
# data = (
#     pd.read_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_sample.csv")
#     .drop(["identifier"], axis=1)
#     .to_numpy()[:1, :]
# )
# %%
# %%
# expression = np.log10(1 + data)
# assert expression.shape[1] == model.config.n_genes
# input_ids = tokenizer.batch_encode_plus(expression, return_tensors="pt")["input_ids"]
# %%
# %%
preds = out["logits"].argmax(dim=-1)
preds[0, 0]
correct = (preds == input_ids).sum()
correct
correct / input_ids.shape[1]
model.config
# %%
