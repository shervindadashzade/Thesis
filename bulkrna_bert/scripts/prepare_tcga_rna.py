# %%
import pandas as pd
import os

df = pd.read_csv("/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/data/tcga_sample.csv")
gene_ids = df.columns[:-1]

tcga_sample_sheet = pd.read_csv(
    "/mnt/hdd/Shervin/Thesis/storage/TCGA/PAN/sample_sheet.tsv", sep="\t"
)
tcga_sample_sheet.columns
tcga_sample_sheet["Tissue Type"].value_counts()
tcga_sample_sheet = tcga_sample_sheet[["File ID", "File Name", "Sample ID"]]
tcga_sample_sheet


tcga_sample_sheet["Sample ID"].map(lambda x: x.split("-")[-1][:-1]).value_counts()

base_dir = "/mnt/hdd/Shervin/Thesis/storage/TCGA/PAN/gene_expressions/data"

tcga_sample_sheet.columns
# %%
tcga_sample_sheet["path"] = tcga_sample_sheet.apply(
    lambda x: os.path.join(base_dir, x["File ID"], x["File Name"]), axis=1
)
# %%
mapping_id_to_name = None
gene_ids = None
for i, row in tcga_sample_sheet.iterrows():
    file_path = row["path"]
    gene_expression = pd.read_csv(file_path, sep="\t", header=1).iloc[4:, :][
        ["gene_id", "gene_name", "tpm_unstranded"]
    ]

    if mapping_id_to_name is None:
        mapping_id_to_name = (
            gene_expression[["gene_id", "gene_name"]]
            .set_index("gene_id")
            .to_dict()["gene_name"]
        )

    if gene_ids is None:
        gene_ids = gene_expression["gene_id"].tolist()

    break
gene_expression
# %%
gene_ids.values
# %%
len(expressions)
# %%
expressions_df = pd.DataFrame(expressions, columns=gene_ids)
# %%
expressions_df["sample_id"] = tcga_sample_sheet["Sample ID"]
# %%
expressions_df.to_csv("storage/TCGA/PAN/rna_summary.csv")
# %%
expressions_df
# %%
