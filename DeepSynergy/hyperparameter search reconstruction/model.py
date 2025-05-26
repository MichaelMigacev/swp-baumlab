import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import pickle
import gzip
import numpy as np

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Encoder(nn.Module):
    def __init__(self, input_size, layers, input_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential()
        
        # Add input layer
        self.layers.append(nn.Linear(input_size, layers[0]))
        self.layers.append(nn.ReLU())
        if input_dropout > 0:
            self.layers.append(nn.Dropout(input_dropout))
        
        # Add hidden layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            # Don't add ReLU and dropout after the last layer
            if i < len(layers) - 1:
                self.layers.append(nn.ReLU())
                if hidden_dropout > 0:
                    self.layers.append(nn.Dropout(hidden_dropout))
    
    def forward(self, x):
        return self.layers(x)

from lightning.pytorch.callbacks import EarlyStopping

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, learning_rate, early_stopping=True, patience=5, min_delta=0.01):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.encoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        val_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("val_loss", val_loss)
        return val_loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        test_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.5)
        return optimizer

    def configure_callbacks(self):
        if not self.early_stopping:
            return []
            
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=True,
            mode="min"
        )
        return [early_stop_callback]

def load_data(data_file):
    with gzip.open(data_file, 'rb') as file:
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
    return X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test
