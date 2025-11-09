import torch
from bulkrna_bert.model import BulkRNABert, BulkRNABertConfig
import lightning as L
from torchmetrics import F1Score, Accuracy, Precision, Recall
import torch.nn.functional as F


class BulkRNABertModel(L.LightningModule):
    def __init__(self, lr=1e-4) -> None:
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
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        out = self.forward(input_ids)
        _, _, C = out["logits"].shape
        out = out["logits"].reshape(-1, C)
        labels = labels.flatten()
        loss = F.cross_entropy(
            out,
            labels,
            ignore_index=-100,
            label_smoothing=0.1,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        out = self.forward(input_ids)
        _, _, C = out["logits"].shape
        out = out["logits"].reshape(-1, C)
        labels = labels.flatten()
        loss = F.cross_entropy(
            out,
            labels,
            ignore_index=-100,
            label_smoothing=0.1,
        )
        preds = out.argmax(-1)
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]
        acc = self.acc_metric(preds, labels)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
        torch.save(all_embeddings, "gdsc_embeddings.pt")
        self.embeddings.clear()


if __name__ == "__main__":
    from bulkrna_bert.datamodule import TCGARNASeqDataModule

    datamodule = TCGARNASeqDataModule(
        mask_percentage=0.12,
        batch_size=16,
        path="/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/gdsc_input_ids.pt",
    )

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
