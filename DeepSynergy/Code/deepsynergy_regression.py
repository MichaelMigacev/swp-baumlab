import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

import sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
data_file = 'DeepSynergy/Code/data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)

#Our hyperparameters, the hyperparameter file is purely cosmetic currently
layers = [8182,4096,1] 
epochs = 1000 
act_func = 'relu'
dropout = 0.5 
input_dropout = 0.2
eta = 0.00001 
norm = 'tanh' 

#We open the normalized dataset
print("Opening file..")
file = gzip.open(data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()

y_tr.shape
X_tr.shape

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(X_tr.shape[1], layers[0]), 
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2]),
                               )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        val_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = F.mse_loss(z,y.view(x.size(0), -1))
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta, momentum=0.5)
        return optimizer

print("Loading dataset..")
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

print("Applying dataset..")
train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

model = LitAutoEncoder(Encoder())

import multiprocessing
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

num_workers = multiprocessing.cpu_count() - 1

# Logging + Callbacks einrichten
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=True,
    mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
    filename="best_model-{epoch:02d}-{val_loss:.4f}"
)

logger = CSVLogger("lightning_logs", name="deepsynergy")

# Dataloader
print("Loading data into model..")
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=num_workers)

# Trainer
print("Training starts..")
trainer = L.Trainer(
    max_epochs=1000,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    log_every_n_steps=5
)

# Start Training
trainer.fit(model, train_loader, valid_loader)

# --------------------------------------
# Analyse the val_loss with Moving Average
# --------------------------------------
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Load LogFlie from Csv Logger
metrics_path = os.path.join(logger.log_dir, "metrics.csv")
metrics_df = pd.read_csv(metrics_path)

# Calc Moving Average val_loss 
val_loss = metrics_df["val_loss"].dropna().values
average_over = 15
mov_av = moving_average(val_loss, average_over)
smooth_val_loss = np.pad(mov_av, int(average_over / 2), mode='edge')
best_epoch = np.argmin(smooth_val_loss)
print(f"Beste Epoche basierend auf smoothed validation loss: {best_epoch}")

# Plot for Visualization
plt.figure(figsize=(16, 8))
plt.plot(val_loss, label='Validation Loss')
plt.plot(smooth_val_loss, label='Smoothed Validation Loss', linewidth=2)
plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss & Smoothed Curve")
plt.grid(True)
plt.show()

### Wir brauchen das nicht, da wir die besten Hyperparameter bereits kennen ###
# Optional: Modell bis zur besten Epoche erneut trainieren (wie im Keras-Notebook)
# trainer = L.Trainer(max_epochs=best_epoch, ...)
# trainer.fit(model, DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=64), DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=64))

# Modell nach bestem Training testen
trainer.test(model, dataloaders=DataLoader(test_dataset))

# Vorhersagen erzeugen
with torch.no_grad():
    y_pred = model.encoder(X_test).detach()


