import pandas as pd
import numpy as np

csv_path = "/mnt/hdd/Shervin/Thesis/storage/GDSC/rna_seq_passports/rnaseq_merged_rsem_tpm_20250922.csv"


df = pd.read_csv(csv_path)
df
# %%
model_names = df.iloc[0, :][3:]
df = df.iloc[3:, :]
# %%
# rows are genes and columns are samples
expressions = df.iloc[:, 3:].values

expressions = expressions.astype(np.float32)
expressions = np.nan_to_num(expressions, nan=0.0)


ref_gene_ids = (
    pd.read_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_sample.csv")
    .columns[:-1]
    .tolist()
)

gdsc_genes = df.iloc[:, 1].tolist()
# %%
expression_values = np.zeros((len(ref_gene_ids), len(model_names)))
for idx, gene in enumerate(ref_gene_ids):
    if gene not in gdsc_genes:
        continue
    gene_idx = gdsc_genes.index(gene)
    expression_values[idx, :] = expressions[gene_idx, :]
# %%
expression_values


gdsc_expression = pd.DataFrame(data=expression_values.T, columns=ref_gene_ids)
gdsc_expression["identifier"] = model_names.values.tolist()


gdsc_expression.to_csv("bulkrna_bert/data/gdsc_all.csv")

from bulkrna_bert.tokenizer import BinnedOmicTokenizer

repo = "InstaDeepAI/BulkRNABert"
tokenizer = BinnedOmicTokenizer.from_pretrained(repo)

expressions = gdsc_expression.drop(["identifier"], axis=1).values

inputs = tokenizer.batch_encode_plus(expressions, return_tensors="pt")
input_ids = inputs["input_ids"]

import torch

input_ids = torch.load("bulkrna_bert/data/gdsc_input_ids.pt")
input_ids[input_ids == 64] = 63
torch.save(input_ids, "bulkrna_bert/data/gdsc_input_ids.pt")
