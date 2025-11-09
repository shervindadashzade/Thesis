from lightning.pytorch import callbacks
from bulkrna_bert.datamodule import TCGARNASeqDataModule
from bulkrna_bert.module import BulkRNABertModel
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

seed_everything(42, workers=True)

datamodule = TCGARNASeqDataModule(
    path="/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt",
    batch_size=11,
    mask_percentage=0.12,
    only_mask=False,
)

model = BulkRNABertModel(lr=1e-5)


tf_logger = TensorBoardLogger(save_dir="tf_logs", name="bulkrna_bert_fine_tune_gdsc")
early_stop_cb = EarlyStopping("val_acc", patience=10, mode="max")
best_val_loss_cb = ModelCheckpoint(
    "checkpoints", "best_loss_{epoch}_{val_loss:.4f}", "val_loss"
)
best_val_acc_cb = ModelCheckpoint(
    "checkpoints", "best_acc_{epoch}_{val_acc:.4f}", "val_acc", mode="max"
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    precision="bf16-mixed",
    max_epochs=100,
    log_every_n_steps=1,
    accumulate_grad_batches=16,
    gradient_clip_val=1,
    deterministic=True,
    logger=tf_logger,
    callbacks=[best_val_loss_cb, best_val_acc_cb, early_stop_cb],
)

trainer.fit(model, datamodule)
