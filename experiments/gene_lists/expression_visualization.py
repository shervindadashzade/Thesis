#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
# %%
base_dir = 'storage/processed'

with open(os.path.join(base_dir, 'gdsc.pkl'),'rb') as f:
    gdsc_data = pickle.load(f)

with open(os.path.join(base_dir, 'ccle.pkl'),'rb') as f:
    ccle_data = pickle.load(f)

with open(os.path.join(base_dir, 'tcga.pkl'),'rb') as f:
    tcga_data = pickle.load(f)
# %%
gene_names = gdsc_data['gene_names']
#%%
tcga_expression = tcga_data['expression'].T
gdsc_expression = gdsc_data['expression'].T
ccle_expression = ccle_data['expression'].T
# %%
tcga_rnd_indices = np.random.choice(np.arange(tcga_expression.shape[0]), size=5000)
tcga_expression_rnd = tcga_expression[tcga_rnd_indices, :]
# %%
all_expressions = np.concat([ccle_expression, gdsc_expression, tcga_expression_rnd], axis=0)
# %%
all_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(all_expressions)
ccle_embeddings = all_embeddings[:ccle_expression.shape[0],:]
gdsc_embeddings = all_embeddings[ccle_expression.shape[0]:ccle_expression.shape[0]+gdsc_expression.shape[0],:]
tcga_embeddings = all_embeddings[ccle_expression.shape[0]+gdsc_expression.shape[0]:,:]
# %%
plt.figure(figsize=(15,10))
plt.scatter(ccle_embeddings[:,0], ccle_embeddings[:,1],s=3, label='CCLE',alpha=0.5)
plt.scatter(gdsc_embeddings[:,0], gdsc_embeddings[:,1],s=3, label='GDSC', alpha=0.5)
plt.scatter(tcga_embeddings[:,0], tcga_embeddings[:,1],s=3, label='TCGA', alpha=0.5)
plt.legend()
plt.title('Log1p Transformed Expressions')
plt.show()
# %%
ccle_expression_normalized = StandardScaler().fit_transform(ccle_expression)
gdsc_expression_normalized = StandardScaler().fit_transform(gdsc_expression)
tcga_expression_normalized = StandardScaler().fit_transform(tcga_expression_rnd)
all_expressions_normalized = np.concat([ccle_expression_normalized, gdsc_expression_normalized, tcga_expression_normalized], axis=0)
#%%
all_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(all_expressions_normalized)
ccle_embeddings = all_embeddings[:ccle_expression.shape[0],:]
gdsc_embeddings = all_embeddings[ccle_expression.shape[0]:ccle_expression.shape[0]+gdsc_expression.shape[0],:]
tcga_embeddings = all_embeddings[ccle_expression.shape[0]+gdsc_expression.shape[0]:,:]
# %%
plt.figure(figsize=(15,10))
plt.scatter(ccle_embeddings[:,0], ccle_embeddings[:,1],s=3, label='CCLE',alpha=0.5)
plt.scatter(gdsc_embeddings[:,0], gdsc_embeddings[:,1],s=3, label='GDSC', alpha=0.5)
plt.scatter(tcga_embeddings[:,0], tcga_embeddings[:,1],s=3, label='TCGA', alpha=0.5)
plt.legend()
plt.title('Z-Score Normalized Seperately Log1p Transformed Expressions')
plt.show()
# %%
all_expressions = np.concat([ccle_expression, gdsc_expression, tcga_expression_rnd], axis=0)
all_expressions_normalized = StandardScaler().fit_transform(all_expressions)
# %%
all_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(all_expressions_normalized)
ccle_embeddings = all_embeddings[:ccle_expression.shape[0],:]
gdsc_embeddings = all_embeddings[ccle_expression.shape[0]:ccle_expression.shape[0]+gdsc_expression.shape[0],:]
tcga_embeddings = all_embeddings[ccle_expression.shape[0]+gdsc_expression.shape[0]:,:]
# %%
plt.figure(figsize=(15,10))
plt.scatter(ccle_embeddings[:,0], ccle_embeddings[:,1],s=3, label='CCLE',alpha=0.5)
plt.scatter(gdsc_embeddings[:,0], gdsc_embeddings[:,1],s=3, label='GDSC', alpha=0.5)
plt.scatter(tcga_embeddings[:,0], tcga_embeddings[:,1],s=3, label='TCGA', alpha=0.5)
plt.legend()
plt.title('Z-Score Normalized Log1p Transformed Expressions')
plt.show()
# %%
all_expressions_pca = PCA(n_components=5, random_state=0).fit_transform(all_expressions)
all_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(all_expressions_pca)
ccle_embeddings = all_embeddings[:ccle_expression.shape[0],:]
gdsc_embeddings = all_embeddings[ccle_expression.shape[0]:ccle_expression.shape[0]+gdsc_expression.shape[0],:]
tcga_embeddings = all_embeddings[ccle_expression.shape[0]+gdsc_expression.shape[0]:,:]
# %%
plt.figure(figsize=(15,10))
plt.scatter(ccle_embeddings[:,0], ccle_embeddings[:,1],s=3, label='CCLE',alpha=0.5)
plt.scatter(gdsc_embeddings[:,0], gdsc_embeddings[:,1],s=3, label='GDSC', alpha=0.5)
plt.scatter(tcga_embeddings[:,0], tcga_embeddings[:,1],s=3, label='TCGA', alpha=0.5)
plt.legend()
plt.title('PCA Log1p Transformed Expressions')
plt.show()
# %%
tcga_expression.mean(axis=1)
#%%
gdsc_expression.mean(axis=1)
#%%
ccle_expression.mean(axis=1)
# %%
tcga_expression.std(axis=1)
# %%
gdsc_expression.std(axis=1)
# %%
ccle_expression.std(axis=1)
# %%
plt.figure(figsize=(20,5))
gene_id = np.random.randint(0,gdsc_expression.shape[1],1).item()
plt.suptitle(f'Gene {gene_names[gene_id]} Distribution across datasets')
plt.subplot(1,3,1)
plt.hist(ccle_expression[:,gene_id], bins=100)
plt.title('CCLE')
plt.subplot(1,3,2)
plt.hist(gdsc_expression[:,gene_id], bins=100)
plt.title('GDSC')
plt.subplot(1,3,3)
plt.hist(tcga_expression[:,gene_id], bins=100)
plt.title('TCGA')
plt.show()
# %%
# experiment with most aligned genes distributionally
from scipy.stats import ks_2samp, wasserstein_distance

rng = np.random.default_rng(0)
n = min(len(gdsc_expression), len(tcga_expression))
idx_tcga = rng.choice(len(tcga_expression), size=n, replace=False)
# %%
A = gdsc_expression
B = tcga_expression[idx_tcga]
# %%
bins = np.linspace(0,17, 100)

def js_div(p,q, eps=1e-12):
    p = p / (p.sum() + eps); q= q/(q.sum() + eps)
    m = 0.5 * (p+q)
    def kl(u,v): return np.sum(u * (np.log(u+eps) - np.log(v+eps)))
    return 0.5 * kl(p,m) + 0.5 * kl(q,m)

ks_stats, emd_vals, js_vals = [],[],[]

for g in tqdm(range(A.shape[1])):
       a,b = A[:,g], B[:,g]

       ks = ks_2samp(a,b, alternative='two-sided').statistic

       emd = wasserstein_distance(a,b)

       pa,_ = np.histogram(a, bins=bins)
       pb,_ = np.histogram(b, bins=bins)
       js = js_div(pa.astype(float), pb.astype(float))
       ks_stats.append(ks); emd_vals.append(emd); js_vals.append(js)

ks_stats = np.array(ks_stats); emd_vals = np.array(emd_vals); js_vals = np.array(js_vals)
# %%
def to_similarity(d):
    lo, hi = np.percentile(d, [5, 95])
    d_clip = np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)
    return 1.0 - d_clip

S = (to_similarity(ks_stats) + to_similarity(emd_vals) + to_similarity(js_vals)) / 3.0
# %%
plt.hist(S)
# %%
gene_indices = np.where(S>0.97)[0]
#%%
all_expressions = np.concat([ccle_expression[:,gene_indices], gdsc_expression[:,gene_indices], tcga_expression_rnd[:, ]], axis=0)
# %%
all_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(all_expressions)
ccle_embeddings = all_embeddings[:ccle_expression.shape[0],:]
gdsc_embeddings = all_embeddings[ccle_expression.shape[0]:ccle_expression.shape[0]+gdsc_expression.shape[0],:]
tcga_embeddings = all_embeddings[ccle_expression.shape[0]+gdsc_expression.shape[0]:,:]
# %%
plt.figure(figsize=(15,10))
plt.scatter(ccle_embeddings[:,0], ccle_embeddings[:,1],s=3, label='CCLE',alpha=0.5)
plt.scatter(gdsc_embeddings[:,0], gdsc_embeddings[:,1],s=3, label='GDSC', alpha=0.5)
plt.scatter(tcga_embeddings[:,0], tcga_embeddings[:,1],s=3, label='TCGA', alpha=0.5)
plt.legend()
plt.title('Log1p Transformed Expressions')
plt.show()