#%%
import json
import os.path as osp
from goatools.anno.gaf_reader import GafReader
# %%
gaf_reader = GafReader(osp.join('storage','ontology','goa_human.gaf'))
# %%
ns2assc = gaf_reader.get_ns2assc()
# %%
with open(osp.join('storage','uniprot_to_tcga.json'),'r') as f:
    uniprot_to_tcga = json.load(f)
# %%
uniprot_ids = list(uniprot_to_tcga.keys())
# %%
# first try with amigo2 and only BP terms
all_go_terms = []
genes_with_no_go_term = []
for idx, protein in enumerate(uniprot_ids):
    try:
        all_go_terms += list(ns2assc['BP'][protein])
    except:
        genes_with_no_go_term.append(protein)
# %%
all_go_terms = list(set(all_go_terms))
# %%
from goatools.obo_parser import GODag
godag = GODag(osp.join('storage','ontology','go-basic.obo'))
# %%
# %%
def get_lower_depth_terms(go_id, desired_depth=2):
    visited = []
    to_visit = []
    found_terms = []
    to_visit.append(go_id)
    while len(to_visit):
        go_id = to_visit.pop()
        if go_id not in visited:
            go_term = godag[go_id]
            if go_term.depth <= desired_depth:
                found_terms.append(go_id)
                visited.append(go_id)
                for parent in go_term.parents:
                    to_visit.append(parent.id)
            else:
                for parent in go_term.parents:
                    to_visit.append(parent.id)
                visited.append(go_id)
    return found_terms
# %%
reduced_go_terms = []
for term in all_go_terms:
    reduced_go_terms += get_lower_depth_terms(term, desired_depth=2)
# %%
reduced_go_terms = list(set(reduced_go_terms))
# %%
from collections import Counter
# %%
Counter(list(map(lambda x: godag[x].depth, reduced_go_terms)))
# %%
import networkx as nx
G = nx.Graph()
# %%
for go_id in reduced_go_terms:
    term = godag[go_id]
    G.add_node(go_id, name=term.name)

rows = []
for go_id in reduced_go_terms:
    term = godag[go_id]
    for parent in term.parents:
        if parent.id in reduced_go_terms:
            G.add_edge(parent.id, go_id)
            rows.append((term.name,parent.name,'is_a'))

# %%
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=False,node_size=10)
# %%
import pandas as pd
# %%
rows

# %%
pd.DataFrame(rows,columns=['source','target','rel']).to_csv('storage/test.csv')
# %%
root = 'GO:0008150'
# %%
root = godag[root]
# %%
rows = []
rows_att = []
for child in root.children:
    rows.append((root.name, child.name, 'is_a'))
    rows_att.append((root.name,1))
    for child_child in child.children:
        rows.append((child.name, child_child.name,'is_a'))
        rows_att.append((child.name,2))
# %%
pd.DataFrame(rows,columns=['source','target','rel']).to_csv('storage/test.csv')
# %%
pd.DataFrame(rows_att,columns=['source','target','rel']).to_csv('storage/test.csv')