# %%
import pandas as pd
import numpy as np


csv_path = "/mnt/hdd/Shervin/Thesis/storage/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

df = pd.read_csv(csv_path, index_col=0)

df

model_ids = df["ModelID"].tolist()

df

import mygene

mg = mygene.MyGeneInfo()

ccle_genes = df.columns[5:]


# %%
def get_ensemble_id(gene_id):
    ensemble_id = mg.query(gene_id, fields="ensembl.gene", species="human")["hits"][0][
        "ensembl"
    ]["gene"]
    return ensemble_id


# %%

ccle_gene_entrez_ids = list(map(lambda x: x.split()[1][1:-1], ccle_genes))

result = mg.querymany(
    ccle_gene_entrez_ids, scopes=["entrezgene"], fields="ensembl.gene", species="human"
)
# %%
from tqdm import tqdm

entrez_id_to_ensemble_id = {}

for res in tqdm(result):
    entrez_id = res["query"]
    if "ensembl" in res:
        ensemble_id = res["ensembl"]
        if isinstance(ensemble_id, list):
            ensemble_id = ensemble_id[0]
        ensemble_id = ensemble_id["gene"]
        entrez_id_to_ensemble_id[entrez_id] = ensemble_id
    else:
        print("skipping the gene entrez_id=", entrez_id)
# %%
#
entrez_id_to_ensemble_id

ccle_gene_ids = list(
    map(
        lambda x: entrez_id_to_ensemble_id[x]
        if x in entrez_id_to_ensemble_id
        else None,
        ccle_gene_entrez_ids,
    )
)
# %%
len(ccle_gene_ids)
# %%
ref_gene_ids = (
    pd.read_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_sample.csv")
    .columns[:-1]
    .tolist()
)

len(ref_gene_ids)

common_gene_ids = set(ref_gene_ids).intersection(set(ccle_gene_ids))

len(common_gene_ids)


# %%
expressions = np.zeros((len(model_ids), len(ref_gene_ids)))
ccle_expressions = df.iloc[:, 5:].values

for idx, gene_id in enumerate(ref_gene_ids):
    if gene_id in ccle_gene_ids:
        gene_idx = ccle_gene_ids.index(gene_id)
        expressions[:, idx] = ccle_expressions[:, gene_idx]
# %%

from bulkrna_bert.tokenizer import BinnedOmicTokenizer

repo = "InstaDeepAI/BulkRNABert"
tokenizer = BinnedOmicTokenizer.from_pretrained(repo)
# %%
expressions.max()
inputs = tokenizer.batch_encode_plus(expressions, return_tensors="pt")
# %%
input_ids = inputs["input_ids"]
input_ids[input_ids == 64] = 63

# %%
ccle_df = pd.DataFrame(data=expressions, columns=ref_gene_ids)
ccle_df["identifier"] = model_ids
ccle_df

ccle_df.to_csv("bulkrna_bert/data/ccle_all.csv")


input_ids.max()
import torch

torch.save(input_ids, "bulkrna_bert/data/ccle_input_ids.pt")
