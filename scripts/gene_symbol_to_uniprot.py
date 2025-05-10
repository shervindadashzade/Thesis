import pandas as pd
import os.path as osp
from bioservices import UniProt
from tqdm import tqdm
from helper import Logger
import json
import pickle

df = pd.read_csv('/home-old/shervin/datasets/LUAD/RNA-Seq/0b73f593-b6e8-4897-9b9b-f09e65e436b7/30a6ea89-8eda-47cf-b336-e070fcd08740.rna_seq.augmented_star_gene_counts.tsv', delimiter='\t',skiprows=1)

logger = Logger('logs/gene_symbol_to_uniprot_scripts.logs', append=False)

tcga_gene_names = df[df['gene_type'] == 'protein_coding']['gene_name'].unique()[1:]

uniprot = UniProt()
uniprot_to_tcga = {}
failed_genes = []
i = 0
for idx, gene_name in tqdm(enumerate(tcga_gene_names),total=len(tcga_gene_names)):
    try:
        query = f'(gene:{gene_name}) AND (organism_id:9606)'
        res = uniprot.search(query, limit=2, frmt='json')
        uniprot_id = res['results'][0]['primaryAccession']
        uniprot_to_tcga[uniprot_id] = gene_name
    except Exception as e:
        logger.log(f"For {gene_name} couldn't find uniprot id due to {e}")
        failed_genes.append(gene_name)
        i+=1
logger.log(f'Failure to find Swiss prot for #{i} genes')

logger.log('Writing uniprot_to_tcga.json....')
with open('storage/uniprot_to_tcga.json','w') as f:
    json.dump(uniprot_to_tcga, f)
logger.log('Writing failed genes...')
with open('storage/uniprot_script_failed_genes.pkl','wb') as f:
    pickle.dump(failed_genes, f)
