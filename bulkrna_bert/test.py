from bulkrna_bert.module import BulkRNABertModel
from bulkrna_bert.datamodule import TCGARNASeqDataModule
import lightning as L

# %%
model = BulkRNABertModel.load_from_checkpoint(
    "/mnt/hdd/Shervin/Thesis/checkpoints/best_acc_epoch=99_val_acc=0.4107.ckpt",
)

# input_ids_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt"
# input_ids_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_input_ids.pt"
input_ids_path = "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/ccle_input_ids.pt"


datamodule = TCGARNASeqDataModule(
    path=input_ids_path,
    batch_size=16,
    mask_percentage=0.12,
    only_mask=True,
)

trainer = L.Trainer(accelerator="gpu", devices=[0])

trainer.test(model, datamodule)
