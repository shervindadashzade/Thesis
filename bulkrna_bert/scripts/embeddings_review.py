# %%
import torch
import plotext as plt
from sklearn.manifold import TSNE

tcga_embeddings = torch.load(
    "/mnt/hdd/Shervin/Thesis/bulkrna_bert/embeddings/tcga_without_masking_embeddings.pt"
)
gdsc_embeddings = torch.load(
    "/mnt/hdd/Shervin/Thesis/bulkrna_bert/embeddings/gdsc_without_masking_embeddings.pt"
)

tsne = TSNE(n_components=2, perplexity=30)

N = tcga_embeddings.shape[0]
all_data = torch.concat([tcga_embeddings, gdsc_embeddings], dim=0)

all_tsne = tsne.fit_transform(all_data)

tcga_tsne = all_tsne[:N, :]
gdsc_tsne = all_tsne[N:, :]

tcga_tsne.shape
# %%
plt.scatter(tcga_tsne[:, 0], tcga_tsne[:, 1], label="tcga")
plt.scatter(gdsc_tsne[:, 0], gdsc_tsne[:, 1], label="gdsc")
plt.show()
plt.clear_figure()
# %%
