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


train_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype2/storage/data/train.pkl')
val_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype2/storage/data/val.pkl')
test_dataset = GeneMLMDataset('/mnt/hdd/Shervin/Thesis/prototype2/storage/data/test.pkl')


train_loader = DataLoader(train_dataset, batch_size=8, num_workers=55)
val_loader = DataLoader(val_dataset, batch_size=8)
# test_loader = DataLoader(test_dataset, batch_size=8, num_workers=55)

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

model = GeneEncoder(num_genes=NUM_GENE, load_g2v=True, vocab_size=VOCAB_SIZE, d_model=200, n_heads=10, depth=6, dropout=0.1, mlp_ratio=4.0, activation=nn.GELU, lr=1e-4)

logger = TensorBoardLogger('logs', name='ExpMLMTraining')

checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}-{val_accuracy:.2f}-best_acc.pt', monitor='val_accuracy',mode='max')

checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}-best_loss.pt', monitor='val_loss',mode='min')

trainer = L.Trainer(accelerator='gpu', devices=[0], precision='bf16', fast_dev_run=False, gradient_clip_val=1, accumulate_grad_batches=32, max_epochs=100, val_check_interval=0.1, logger=logger, log_every_n_steps=1)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path='/mnt/hdd/Shervin/Thesis/prototype2/logs/ExpMLMTraining/version_0/checkpoints/epoch=0-step=32.ckpt')
