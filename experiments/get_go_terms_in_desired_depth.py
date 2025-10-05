#%%
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
import json
import matplotlib.pyplot
import networkx as nx
from tqdm import tqdm
%matplotlib tk
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
#%%
godag = GODag('storage/ontology/go-basic.obo')
#%%
ogaf = GafReader('storage/ontology/goa_human.gaf')
#%%
ns2assc = ogaf.get_ns2assc()
#%%
with open('storage/uniprot_to_tcga.json','r') as f:
    uniprot_to_tcga = json.load(f)
tcga_to_uniprot = {}
for key,value in uniprot_to_tcga.items():
    tcga_to_uniprot[value] = key
#%%
def get_lower_depth_terms(go_id, desired_depth=2):
    visited = []
    to_visit = []
    found_terms = []
    to_visit.append(go_id)
    while len(to_visit):
        go_id = to_visit.pop()
        if go_id not in visited:
            go_term = godag[go_id]
            if go_term.depth == desired_depth:
                found_terms.append(go_id)
                visited.append(go_id)
            else:
                for parent in go_term.parents:
                    to_visit.append(parent.id)
                visited.append(go_id)
    return found_terms
#%%
# experimenting with depth of 2
import numpy as np
import pandas as pd
# %%
df = pd.read_csv('storage/epxression_tpm.csv')
# %%
df.head()
# %%
# just starting from bp terms
gene_go_terms = {}
for gene in df.columns[:-1]:
    try:
        gene_uniprot_id = tcga_to_uniprot[gene]
    except:
        continue
    try:
        bp_terms = ns2assc['BP'][gene_uniprot_id]
    except:
        continue
    gene_go_terms[gene] = list(bp_terms)
# %%
len(gene_go_terms)
# %%
# map GO terms to depth 2
gene_reduced_go_terms = {}
for gene, terms in gene_go_terms.items():
    new_terms = []
    for term in terms:
        reduced_depth_term = get_lower_depth_terms(term,desired_depth=2)
        new_terms += reduced_depth_term
    new_terms = list(set(new_terms))
    gene_reduced_go_terms[gene] = new_terms
# %%
len(gene_reduced_go_terms)
# %%
gene_reduced_go_terms
# %%
depth_2_terms = []
for gene, terms in tqdm(gene_reduced_go_terms.items()):
    for term in terms:
        if godag[term].depth == 2:
            depth_2_terms.append(term)
        else:
            print(f'Term {term} is in depth {godag[term].depth}')
# %%
depth_2_terms = list(set(depth_2_terms))
#%%
def random_color():
    return random.choice(list(mcolors.CSS4_COLORS.values()))
#%%
G = nx.Graph()
# visualizing GO terms till depth 2
for go_term in depth_2_terms:
    visited = []
    to_visit = []
    to_visit.append(go_term)
    while len(to_visit):
        go_term = to_visit.pop()
        go_name = godag[go_term].name
        for parent in godag[go_term].parents:
                if parent.depth == godag[go_term].depth - 1:
                    to_visit.append(parent.id)
                    G.add_node(go_name,type='go_term')
                    G.add_node(parent.name, type='go_term')
                    G.add_edge(go_name, parent.name, color=)
#%%
i = 0
for gene, terms in gene_reduced_go_terms.items():
    if i == 100:
        break
    G.add_node(gene, type='gene')
    for term in terms:
        G.add_edge(gene,godag[term].name)
    i += 1
# %%
colors = ['blue' if G.nodes[n]['type'] == 'go_term' else 'yellow' for n in G.nodes()]
sizes = [100 if G.nodes[n]['type'] == 'go_term' else 60 for n in G.nodes()]
#%%
%matplotlib tk
#nx.draw_spring(G, with_labels=True,node_size=sizes,font_size=6,edge_color='red', node_color=colors)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)
nx.draw_networkx_edges(G,pos, edge_color='black', alpha=0.5)
nx.draw_networkx_labels(G,pos,font_size=6)
matplotlib.pyplot.show()
# %%
len(G.nodes)
# %%
gene_reduced_go_terms