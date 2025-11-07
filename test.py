import torch
from bulkrna_bert.model import BulkRNABert, BulkRNABertConfig
from bulkrna_bert.tokenizer import BinnedOmicTokenizer
from bulkrna_bert.datamodule import RNASeqDataset
from torch.utils.data import DataLoader

repo = "InstaDeepAI/BulkRNABert"
config = BulkRNABertConfig.from_pretrained(repo)
model = BulkRNABert.from_pretrained(repo, config=config)
tokenizer = BinnedOmicTokenizer.from_pretrained(repo)
device = "cuda:0"
model.eval()
model.to(torch.bfloat16)
model.to(device)

dataset = RNASeqDataset()
loader = DataLoader(dataset, batch_size=16)
batch = next(iter(loader))
batch
input_ids = batch["input_ids"].to(device)
input_ids.shape

with torch.no_grad():
    out = model(input_ids)

preds = out["logits"].argmax(dim=-1).cpu()
labels = batch["labels"]
mask = labels != -100
total = mask.sum()
correct = (labels[mask] == preds[mask]).sum()
correct / total
# %%
config.n_genes
model
model.gene_embedding_layer
input_ids.shape
# %%
model.expression_embedding_layer

input_ids.max()
# %%
tokenizer.vocab
