import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import pickle, gzip

# --------------------------------
# 1. Daten laden
# --------------------------------
def load_data(path="test0val1normtanh_norm.p.gz"):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

# --------------------------------
# 2. Einfacher Datamodule
# --------------------------------
class SynergyDataModule(L.LightningDataModule):
    def __init__(self, path, batch_size=128):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def prepare_data(self):
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = load_data(self.path)
        self.train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.float32))
        self.val = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                 torch.tensor(y_val, dtype=torch.float32))
        self.test = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

# --------------------------------
# 3. Einfaches tiefes Modell
# --------------------------------
class DeepSynergyNet(L.LightningModule):
    def __init__(self, input_size, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, _):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --------------------------------
# 4. Training + Save Results
# --------------------------------
if __name__ == "__main__":
    import os

    path = "test0val1normtanh_norm.p.gz"
    results_dir = "results_dnn"
    os.makedirs(results_dir, exist_ok=True)

    dm = SynergyDataModule(path)
    dm.prepare_data()
    input_size = dm.train.tensors[0].shape[1]

    model = DeepSynergyNet(input_size)

    trainer = L.Trainer(max_epochs=50, accelerator="auto")
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # Save model
    model_path = os.path.join(results_dir, "deep_synergy_model.ckpt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save test predictions
    model.eval()
    test_loader = dm.test_dataloader()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            predictions.append(pred)
            targets.append(y)

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    # Save predictions and targets
    results_path = os.path.join(results_dir, "test_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({"predictions": predictions, "targets": targets}, f)
    print(f"Test results saved to {results_path}")

