# %%
from transformers import AutoTokenizer
from bulkrna_bert.model import BulkRNABertConfig, BulkRNABert
import pandas as pd
import numpy as np
import torch

# %%
repo = "InstaDeepAI/BulkRNABert"
# %%
config = BulkRNABertConfig.from_pretrained(repo)
config.embeddings_layers_to_save = (4,)
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
# %%
model = BulkRNABert.from_pretrained(repo, config=config)
# %%
csv_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/data/tcga_sample.csv"

gene_expression_array = (
    pd.read_csv(csv_path).drop(["identifier"], axis=1).to_numpy()[:1, :]
)
gene_expression_array = np.log10(1 + gene_expression_array)
assert gene_expression_array.shape[1] == config.n_genes
# %%
device = "cuda"
model = model.to(device).to(torch.bfloat16)
# %%
gene_expression_ids = tokenizer.batch_encode_plus(
    gene_expression_array, return_tensors="pt"
)["input_ids"]
gene_expression_ids = gene_expression_ids.to(device)
# %%
embeddings = model(gene_expression_ids)
# %%
embeddings.keys()
embeddings["embeddings_4"].shape
type(tokenizer)
# %%
from bulkrna_bert.tokenizer import BinnedOmicTokenizer
import pandas as pd
import numpy as np

tokenizer: BinnedOmicTokenizer = BinnedOmicTokenizer.from_pretrained(repo)

data = pd.read_csv(
    "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_all.csv", index_col=0
)

values = data.values[:, :-1].astype(np.float32)
values = np.log10(values + 1)
c = values[:2, :]
input_ids = tokenizer.batch_encode_plus(c, return_tensors="pt")
input_ids["input_ids"]

mask_token_id = tokenizer.vocab[tokenizer.mask_token]

del data

(input_ids["input_ids"][0] == 65).sum()
# %%

a = input_ids["input_ids"][0].clone()
a

import torch
import math

mask_percentage = 0.15
num_selected_tokens = math.ceil(mask_percentage * a.shape[0])
selected_indices = torch.randperm(a.shape[0])[:num_selected_tokens]
selected_indices.shape
len_mask = math.ceil(0.8 * num_selected_tokens)
len_no_change = int(0.1 * num_selected_tokens)
len_random_change = num_selected_tokens - len_mask - len_no_change
print(len_mask, len_no_change, len_random_change)
# %%

mask_indices, no_change_indices, random_change_indices = torch.utils.data.random_split(
    selected_indices,
    [len_mask, len_no_change, len_random_change],
    generator=torch.Generator(),
)
mask_indices = torch.tensor(mask_indices)
no_change_indices = torch.tensor(no_change_indices)
random_change_indices = torch.tensor(random_change_indices)
# %%

a

labels = torch.zeros_like(a) - 100
labels
labels[selected_indices] = a[selected_indices]
labels
selected_indices[0]
input_ids[0]
a[mask_indices] = mask_token_id
a[random_change_indices] = torch.randint(0, 64, (len_random_change,))

torch.where(a == mask_token_id)[0][0]
a[13]
labels[13]
