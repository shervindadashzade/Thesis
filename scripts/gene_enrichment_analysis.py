#%%
from goatools.obo_parser import GODag
#%%
godag = GODag('data/ontology/go-basic.obo')
# %%
from goatools.anno.genetogo_reader import Gene2GoReader
# %%
objanno = Gene2GoReader('data/ontology/gene2go', taxids=[9606])
#%%
go2geneids_human = objanno.get_id2gos(namespace='BP',go2geneids=True)
# %%
from goatools.go_search import GoSearch

srchelp = GoSearch('data/ontology/go-basic.obo',go2items=go2geneids_human)
# %%
import re

cell_cycle_all = re.compile(r'cell cycle', flags=re.IGNORECASE)
cell_cycle_not = re.compile(r'cell cycle.independent', flags=re.IGNORECASE)
# %%
fout_allgos = "cell_cycle_gos_human.log"

with open(fout_allgos, 'w') as log:
    gos_cc_all = srchelp.get_matching_gos(cell_cycle_all, prt=log)
    gos_no_cc = srchelp.get_matching_gos(cell_cycle_not,gos=gos_cc_all,prt=log)

    gos = gos_cc_all.difference(gos_no_cc)

    gos_all = srchelp.add_children_gos(gos)

    geneids = srchelp.get_items(gos_all)

print("{N} human NCBI Entrez GeneIDs related to 'cell cycle' found.".format(N=len(geneids)))
# %%
from goatools.anno.gaf_reader import GafReader
# %%
ogaf = GafReader('data/ontology/goa_human.gaf')
# %%
ns2assc = ogaf.get_ns2assc()
#%%
import pandas as pd
# %%
df = pd.read_csv('/home-old/shervin/datasets/LUAD/RNA-Seq/0b73f593-b6e8-4897-9b9b-f09e65e436b7/30a6ea89-8eda-47cf-b336-e070fcd08740.rna_seq.augmented_star_gene_counts.tsv', delimiter='\t',skiprows=1)
# %%
tcga_gene_names = df[df['gene_type'] == 'protein_coding']['gene_name'].unique()[1:]
# %%
set_anno_gene_names = set(ns2assc['BP'].keys())
# %%
set_tcga_gene_names = set(tcga_gene_names)
# %%
set_anno_gene_names.intersection(set_tcga_gene_names)
# %%
from mygene import MyGeneInfo
# %%
mg = MyGeneInfo()
# %%
res = mg.querymany(set_tcga_gene_names , scopes='symbol', fields='uniprot', species='human')
#%%
uniprot_to_tcga = {}
for entry in res:
    tcga_id = entry['query']
    uniprot_id = entry.get('uniprot',{})
    if 'Swiss-Prot' in uniprot_id:
        uniprot_to_tcga[uniprot_id['Swiss-Prot'][0]] = tcga_id 
# %%
uniprot_to_tcga
# %%
i = 0
for entry in res:
    if 'uniprot' not in entry:
       print(entry)
    i+=1
print(i)
# %%
res = mg.querymany(set_tcga_gene_names , scopes='symbol', fields='uniprot', species='human')
# %%
res
# %%
from bioservices import UniProt

u = UniProt()
# %%
res = u.search("(gene:NUTM2E) AND (organism_id:9606)",frmt='json',limit=2)
print(res['results'][0]['primaryAccession'])
# %%
from tqdm import tqdm
uniprot_to_tcga = {}
i = 0
for gene_name in tqdm(tcga_gene_names):
    try:
        query = f'(gene:{gene_name}) AND (organism_id:9606)'
        res = u.search(query, limit=2, frmt='json')
        uniprot_id = res['results'][0]['primaryAccession']
        uniprot_to_tcga[uniprot_id] = gene_name
    except Exception as e:
        print(f'For {gene_name} couldnt find uniprot id due to {e}')
        i+=1
print(i)
# %%
