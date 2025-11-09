from lightning.pytorch import callbacks
from bulkrna_bert.datamodule import TCGARNASeqDataModule, RNASeqDataModule
from bulkrna_bert.module import BulkRNABertModel
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os

seed_everything(42, workers=True)

# datamodule = TCGARNASeqDataModule(
#     path="/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt",
#     batch_size=11,
#     mask_percentage=0.12,
#     only_mask=False,
# )

datamodule = RNASeqDataModule(batch_size=11)

model = BulkRNABertModel(lr=1e-5)

experiment_name = "bulkrna_bert_fine_tune_all"
checkpoint_dir = os.path.join("checkpoints", experiment_name)

tf_logger = TensorBoardLogger(save_dir="tf_logs", name=experiment_name)
early_stop_cb = EarlyStopping("val_acc", patience=10, mode="max")
best_val_loss_cb = ModelCheckpoint(
    checkpoint_dir, "best_loss_{epoch}_{val_loss:.4f}", "val_loss"
)
best_val_acc_cb = ModelCheckpoint(
    checkpoint_dir, "best_acc_{epoch}_{val_acc:.4f}", "val_acc", mode="max"
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
    val_check_interval=0.2,
)

trainer.fit(model, datamodule)
