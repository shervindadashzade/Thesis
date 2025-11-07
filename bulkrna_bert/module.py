import torch
from bulkrna_bert.model import BulkRNABert, BulkRNABertConfig
import lightning as L
from torchmetrics import F1Score, Accuracy, Precision, Recall


class BulkRNABertModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.repo = "InstaDeepAI/BulkRNABert"
        self.config = BulkRNABertConfig.from_pretrained(self.repo)
        self.config.embeddings_layers_to_save = (4,)
        self.model = BulkRNABert.from_pretrained(self.repo, config=self.config)
        self.f1_metric = F1Score("multiclass", num_classes=64, average="weighted")
        self.acc_metric = Accuracy("multiclass", num_classes=64)
        self.precision_metric = Precision(
            "multiclass", num_classes=64, average="weighted"
        )
        self.recall_metric = Recall("multiclass", num_classes=64, average="weighted")
        self.embeddings = []

    def forward(self, input_ids):
        return self.model(input_ids)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = labels != -100
        out = self.forward(input_ids)

        embeddings = out["embeddings_4"].mean(dim=1)

        if mask.sum() != 0:
            preds = out["logits"].argmax(dim=-1)[mask]
            labels = labels[mask]
            f1 = self.f1_metric(preds, labels)
            acc = self.acc_metric(preds, labels)
            precision = self.precision_metric(preds, labels)
            recall = self.recall_metric(preds, labels)
            self.log_dict(
                {"F1": f1, "Accuracy": acc, "Precision": precision, "Recall": recall},
            )
        self.embeddings.append(embeddings.detach().cpu())

    def on_test_epoch_end(self):
        all_embeddings = torch.concat(self.embeddings, dim=0)
        torch.save(all_embeddings, "tcga_without_masking_embeddings.pt")
        self.embeddings.clear()


if __name__ == "__main__":
    from bulkrna_bert.datamodule import TCGARNASeqDataModule

    datamodule = TCGARNASeqDataModule(mask_percentage=0.0, batch_size=16)
    model = BulkRNABertModel()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        fast_dev_run=False,
        precision="bf16-mixed",
        # limit_test_batches=2,
    )
    trainer.test(model, datamodule)
# %%
