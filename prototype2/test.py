#%%
import lightning as L
import sys
import torch
sys.path.append('/mnt/hdd/Shervin/Thesis')
from prototype2.dataset import GeneMLMDataset
from torch import optim, nn, utils, Tensor
from prototype2.model import FlashSDPA, EncoderBlock, CrossAttentionBlock
import lightning as L
from prototype2.constants import NUM_GENE, VOCAB_SIZE
from torch.utils.data import DataLoader
import pickle
import torchmetrics as metrics
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
import numpy as np
#%%
class GeneEncoder(L.LightningModule):
    def __init__(self, num_genes=NUM_GENE, load_g2v=True, vocab_size = VOCAB_SIZE, depth=3, d_model=200, n_heads=10, dropout=0.0, mlp_ratio=4.0, activation=nn.GELU, lr=1e-4):
        super().__init__()
        self.lr = lr
        if load_g2v:
            with open('/mnt/hdd/Shervin/Thesis/prototype1/storage/data/gene_embeddings.pkl','rb') as f:
                gene_embeddings = torch.from_numpy(pickle.load(f))
                gene_embeddings = torch.concat([torch.zeros(1,gene_embeddings.shape[1]), gene_embeddings], dim=0)
            self.g2v_embedding = nn.Embedding.from_pretrained(gene_embeddings, freeze=True)
        else:
            self.g2v_embedding = nn.Embedding(num_genes, d_model)
        self.vocab_embedding = nn.Embedding(vocab_size, d_model)
        self.depth = depth
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout, mlp_ratio, activation) for i in range(depth)])

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # self.head.weight = self.vocab_embedding.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.accuracy_metric = metrics.Accuracy('multiclass', num_classes=VOCAB_SIZE)
        self._log_hyperparams = True
        self.save_hyperparameters()
    
    def forward(self, input_ids):
        x = self.vocab_embedding(input_ids)
        x += self.g2v_embedding.weight
        out = x
        for layer in self.layers:
            out = layer(out)
        
        out = self.head(out)
        return out
    
    def embed(self, input_ids):
        x = self.vocab_embedding(input_ids)
        x += self.g2v_embedding.weight
        out = x
        for layer in self.layers:
            out = layer(out)
        
        return out
    
    def training_step(self, batch, batch_idx):
        _, loss = self.__common_step__(batch, batch_idx)
        self.log('train_loss',loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx, **kwargs):
        labels = batch['labels']
        logits, loss = self.__common_step__(batch, batch_idx)
        mask = labels!=-100
        labels = labels[mask]
        logits = torch.argmax(logits[mask], dim=-1)
        correct = (labels == logits).sum()
        acc = correct / labels.shape[0]
        self.log_dict({'val_loss': loss, 'val_accuracy': acc}, prog_bar=True, batch_size=labels.shape[0])
    
    def test_step(self, batch, batch_idx, **kwargs):
        labels = batch['labels']
        logits, loss = self.__common_step__(batch, batch_idx)
        mask = labels!=-100
        labels = labels[mask]
        logits = torch.argmax(logits[mask], dim=-1)
        correct = (labels == logits).sum()
        acc = correct / labels.shape[0]
        self.log_dict({'test_loss': loss, 'test_accuracy': acc}, prog_bar=True, batch_size=labels.shape[0])
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def __common_step__(self, batch, batch_idx):
        input_ids = batch["input_ids"].long()
        labels    = batch["labels"].long()
        logits = self.forward(input_ids)
        loss = self.loss_fn(logits.transpose(1,2), labels)

        return logits, loss
        
    def on_train_epoch_end(self):
        print('\n')
        return super().on_train_epoch_end()
# %%
model = GeneEncoder.load_from_checkpoint('prototype2/logs/ExpMLMTraining/version_1/checkpoints/epoch=47-step=2570.ckpt')
# %%
val_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype2/storage/data/val.pkl', do_mask=False)
test_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype2/storage/data/test.pkl', do_mask=False)
# %%
val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)
# %%
trainer = L.Trainer()
trainer.test(model, val_loader)
# %%
trainer.test(model, test_loader)
# %%
device = 'cuda'
model = model.to(device)
model.eval()
cls_embeddings = []
cls_labels = []
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].long().to(device)
        _cls_embeddings = model.embed(input_ids).cpu().numpy()[:,0,:]
        dataset_labels = batch['dataset_labels']
        cls_labels.extend(dataset_labels.tolist())
        cls_embeddings.append(_cls_embeddings)
# %%
cls_embeddings = np.vstack(cls_embeddings)
#%%
cls_labels = np.array(cls_labels)
# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %%
cls_tsne = TSNE().fit_transform(cls_embeddings)
# %%
for i in range(3):
    mask = cls_labels == i
    label = 'CCLE'
    if i == 1:
        label = 'GDSC'
    elif i == 2:
        label = 'TCGA'
    plt.scatter(cls_embeddings[mask, 0], cls_embeddings[mask, 1], label=label, s=4, alpha=1)
plt.legend()
plt.show()
# %%
import math
model = model.to('cpu')
input_ids = val_dataset[0]['input_ids'].unsqueeze(0)
# %%
model.eval()
with torch.no_grad():
    x = model.vocab_embedding(input_ids)
    x += model.g2v_embedding.weight
    out = x
    for idx, layer in enumerate(model.layers):
        if idx == 0:
            out = layer.norm1(x)
            B,Sq, D = out.shape
            Sk = Sq
            H = 10
            Dh = D // H
            mha = layer.mha
            q = mha.w_q(out).view(B, Sq, H, Dh).transpose(1, 2)  # [B, H, Sq, Dh]
            k = mha.w_k(out).view(B, Sk, H, Dh).transpose(1, 2)  # [B, H, Sk, Dh]
            A = torch.softmax(q@k.transpose(3,2) / math.sqrt(Dh), dim=-1)
            break
        else:
            out = layer(out)
            print(f'layer{idx+1} processed')
# %%
A = A.mean(dim=1).squeeze()
# %%
plt.figure(figsize=(20,20))
plt.imshow(A.numpy(), cmap='jet', vmin=A.min().item(), vmax=A.max().item())
plt.colorbar()
plt.show()
#%%
A_hat = A[:20,:20]
plt.imshow(A_hat.numpy(), cmap='jet', vmin=A.min().item(), vmax=A.max().item())
plt.colorbar()
plt.show()
# %%
plt.hist(A.flatten(), bins=100)
# %%
normalized_A = (A - A.min()) / (A.max() - A.min())
# %%
plt.imshow(normalized_A)
# %%
with open('storage/processed/gene_names.pkl', 'rb') as f:
    gene_names = pickle.load(f)
# %%
idx1 = 14479
idx2 = 2081

print(gene_names[idx1])
print(gene_names[idx2])
# %%
for i,j in zip(a,b):
    print(gene_names[i])
    print(gene_names[j])
    print('#'*20)
# %%
torch.argmax(A[0,:])
#%%
gene_names[0]
#%%
gene_names[2234]
# %%
gene_names.index('PGRMC1')
# %%
torch.argmax(A[1901,:])
#%%
gene_names[14817]
# %%
gene_names.index('PGRMC2')
# %%
A[1901, 2234]
# %%
A[1901,:].min()
# %%
plt.hist(A[1901,:], bins=100)
# %%
gene_names.index('TMEM170A')
# %%
plt.hist(A[0,:], bins=100)
# %%
gene_names.index('RTN4')
# %%
A[0, 10579]
# %%
torch.argmax(A[0,:])
# %%
gene_names[2234]