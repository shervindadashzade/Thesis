#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
# %%
with open('storage/tcga_temp/gene_expression/data.pkl', 'rb') as f:
    tcga_data = pickle.load(f)
# %%
with open('storage/gdsc_temp/gene_expression/data.pkl', 'rb') as f:
    gdsc_data = pickle.load(f)
# %%
with open('storage/ensmble_id_to_swiss_prot.pkl','rb') as f:
    mapping = pickle.load(f)
# %%
tcga_log_transformed = np.log1p(tcga_data['data'])
gdsc_log_transformed = np.log1p(gdsc_data['data'])
# %%
# filtering the common genes that are also have SwissProt
tcga_gene_ids = [gene_id.split('.')[0] for gene_id in tcga_data['gene_ids']]
gdsc_gene_ids = list(gdsc_data['gene_ids'])
common_genes = set(tcga_gene_ids).intersection(set(gdsc_gene_ids))
common_genes = [gene_id for gene_id in common_genes if gene_id in mapping]
print(len(common_genes))
# %%
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
from tqdm import tqdm
import os

go_dag = GODag('storage/ontology/go-basic.obo')
bp_dag = {}
for go_id in tqdm(go_dag):
    term = go_dag[go_id]
    if term.namespace == 'biological_process':
        bp_dag[go_id] = term
del go_dag
# %%
gaf_obj = GafReader('storage/ontology/goa_human.gaf')
ns2assc = gaf_obj.get_ns2assc()
del gaf_obj
# %%
levels = {}
for go_id in tqdm(bp_dag):
    term = bp_dag[go_id]
    level = term.level
    level_terms = levels.get(level, [])
    level_terms.append(go_id)
    levels[level] = level_terms
for level in range(len(levels)):
    print(f'level: {level} with {len(levels[level])} go terms.')
# %%
def map_go_term_to_target_level(go_id, target_level=4):
    go_term = bp_dag[go_id]
    current_level = go_term.level
    current_terms = [go_term]
    if current_level > 4:
        while current_level != target_level:
            current_level -= 1
            current_terms = [parent for go_term in current_terms for parent in go_term.parents]
    elif current_level < 4:
        while current_level != target_level:
            current_level += 1
            current_terms = [child for go_term in current_terms for child in go_term.children]
    current_terms = list(set([go_term.id for go_term in current_terms if go_term.level == target_level]))
    return current_terms
# %%
# filtering genes that do not have any go term
print(len(common_genes))
common_genes = [gene_id for gene_id in common_genes if mapping[gene_id] in ns2assc['BP']]
print(len(common_genes))
#%%
target_level = 4
gene_to_go = {}
go_to_gene = {}
for gene_id in tqdm(common_genes):
    prot_id = mapping[gene_id]
    go_ids = np.array(list(ns2assc['BP'][prot_id]))
    levels = np.array([bp_dag[go_id].level for go_id in go_ids])
    has_target_level_annotation = (levels == target_level).sum() > 0
    if has_target_level_annotation:
        go_ids = go_ids[levels == target_level].tolist()
    else:
        max_level_idx = np.argmax(levels)
        go_ids = map_go_term_to_target_level(go_ids[max_level_idx], target_level)
    gene_to_go[gene_id] = go_ids
    for go_id in go_ids:
        genes = go_to_gene.get(go_id,[])
        genes.append(gene_id)
        go_to_gene[go_id] = genes
# %%
for go_id, gene_ids in go_to_gene.items():
    if len(gene_ids) == 0:
        print(go_id, ':', len(gene_ids))
#%%
all_gene_ids = sorted(list(set([gene_id for go_id in go_to_gene for gene_id in go_to_gene[go_id]])))
all_go_terms = sorted(list(set(go_to_gene.keys())))
# %%
import torch
import torch.nn as nn
# %%
linear = nn.Linear(len(all_gene_ids), len(all_go_terms))
# %%
data = {
    'all_gene_ids' : all_gene_ids,
    'all_go_terms' : all_go_terms,
    'mapping': go_to_gene
}
# %%
with open('storage/go_nn/data.pkl','wb') as f:
    pickle.dump(data, f)
# %%
with open('storage/go_nn/data.pkl','rb') as f:
    data = pickle.load(f)
# %%
mask_input_layer = np.zeros( (len(data['all_go_terms']), len(data['all_gene_ids'])), dtype=np.bool)
# %%
for go_id, genes in tqdm(data['mapping'].items()):
    for gene_id in genes:
        go_idx = data['all_go_terms'].index(go_id)
        gene_idx = data['all_gene_ids'].index(gene_id)
        mask_input_layer[go_idx][gene_idx] = 1
# %%
mapping_level_4_to_3 = {}
for go_id in tqdm(data['all_go_terms']):
    go_term = bp_dag[go_id]
    mapping_level_4_to_3[go_id] = [parent.id for parent in go_term.parents if parent.level == 3]
# %%
# healthy check
for go_id, parents in mapping_level_4_to_3.items():
    if len(parents) == 0:
        print('oho')
# %%
all_level_3_go_terms = sorted(list(set([parent_id for go_id in mapping_level_4_to_3 for parent_id in mapping_level_4_to_3[go_id]])))
#%%
mask_level_4_to_3 = np.zeros((len(all_level_3_go_terms), len(data['all_go_terms'])), dtype=np.bool)

for go_id_level_4 in tqdm(mapping_level_4_to_3):
    for go_id_level_3 in mapping_level_4_to_3[go_id_level_4]:
        go_idx_level_3 = all_level_3_go_terms.index(go_id_level_3)
        go_idx_level_4 = data['all_go_terms'].index(go_id_level_4)
        mask_level_4_to_3[go_idx_level_3, go_idx_level_4] = True
# %%
mapping_level_3_to_2 = {}
for go_id in tqdm(all_level_3_go_terms):
    go_term = bp_dag[go_id]
    mapping_level_3_to_2[go_id] = [parent.id for parent in go_term.parents if parent.level == 2]
# %%
# healthy check
for go_id, parents in mapping_level_3_to_2.items():
    if len(parents) == 0:
        print('oho')
# %%
all_level_2_go_terms = sorted(list(set([parent_id for go_id in mapping_level_3_to_2 for parent_id in mapping_level_3_to_2[go_id]])))
# %%
mask_level_3_to_2 = np.zeros((len(all_level_2_go_terms), len(all_level_3_go_terms)), dtype=np.bool)

for go_id_level_3 in tqdm(mapping_level_3_to_2):
    for go_id_level_2 in mapping_level_3_to_2[go_id_level_3]:
        go_idx_level_2 = all_level_2_go_terms.index(go_id_level_2)
        go_idx_level_3 = all_level_3_go_terms.index(go_id_level_3)
        mask_level_3_to_2[go_idx_level_2, go_idx_level_3] = True
# %%
np.sum(mask_input_layer) + np.sum(mask_level_4_to_3) + np.sum(mask_level_3_to_2)
# %%
network_data = {
    'input_gene_ids': data['all_gene_ids'],
    'level_4_go_terms': data['all_go_terms'],
    'level_3_go_terms': all_level_3_go_terms,
    'level_2_go_terms': all_level_2_go_terms,
    'input_mask': mask_input_layer,
    'mask_level_4_to_3': mask_level_4_to_3,
    'mask_level_3_to_2': mask_level_3_to_2
}
# %%
with open('storage/go_nn/network_data.pkl','wb') as f:
    pickle.dump(network_data, f)
# %%
