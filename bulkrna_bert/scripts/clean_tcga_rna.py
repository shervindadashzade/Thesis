# %%
import pandas as pd
import numpy as np


tcga_rna = pd.read_csv(
    "/mnt/hdd/Shervin/Thesis/storage/TCGA/PAN/rna_summary.csv", index_col=0
)
# %%
tcga_rna = tcga_rna.drop("Unnamed: 0", axis=1)
# %%
tcga_rna = tcga_rna.loc[:, tcga_rna.sum() != 0]
# %%
tcga_rna.columns = tcga_rna.columns.map(lambda x: x.split(".")[0])
tcga_rna.to_csv("/mnt/hdd/Shervin/Thesis/storage/TCGA/PAN/rna_summary.csv")
# %%
bulkrna_bert_genes = pd.read_csv(
    "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_sample.csv"
).columns[:-1]

# %%
tcga_rna
# %%
tcga_rna[bulkrna_bert_genes]
# %%
for ref_gene in bulkrna_bert_genes:
    if ref_gene not in tcga_rna.columns:
        tcga_rna[ref_gene] = 0
# %%
tcga_rna_filtered = tcga_rna[bulkrna_bert_genes]
tcga_rna_filtered
tcga_rna_filtered["identifier"] = tcga_rna["sample_id"]
# %%
tcga_rna_filtered

tcga_rna_filtered.to_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_all.csv")
tcga_rna_filtered
# %%
import pandas as pd
import numpy as np

df = pd.read_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_all.csv", index_col=0)
expressions = df.values[:, :-1].astype(np.float32)
normalized_expressions = np.log10(expressions + 1)

# %%
from bulkrna_bert.tokenizer import BinnedOmicTokenizer

repo = "InstaDeepAI/BulkRNABert"
tokenizer: BinnedOmicTokenizer = BinnedOmicTokenizer.from_pretrained(repo)
inputs = tokenizer.batch_encode_plus(normalized_expressions, return_tensors="pt")
inputs = inputs["input_ids"]
import torch

inputs.shape
torch.save(inputs, "bulkrna_bert/data/input_ids.pt")
