import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import pickle
import gzip
import numpy as np


# -------------------------------
# Helper functions
# -------------------------------

def save_model(model, path="model.pt"):
    """Save only the model weights (state_dict)."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(model_class, path="model.pt", *args, **kwargs):
    """
    Load a model of type model_class (with constructor arguments *args/**kwargs)
    and restore its saved weights.
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from: {path}")
    return model


def moving_average(a, n=3):
    """Compute a simple moving average of array a with window size n."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_data(data_file):
    """Load gzip-pickled dataset and return all splits/metadata."""
    with gzip.open(data_file, 'rb') as file:
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names, f_feature_origin, f_feature_group = pickle.load(file)
    return X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names, f_feature_origin, f_feature_group


# -------------------------------
# Models
# -------------------------------

class Encoder(nn.Module):
    def __init__(self, input_size, layers, input_dropout=0.0, hidden_dropout=0.0):
        """
        Flexible feed-forward encoder network.
        - input_size: number of input features
        - layers: list of hidden layer sizes (last element = output size)
        - input_dropout: dropout rate after input layer
        - hidden_dropout: dropout rate after hidden layers
        """
        super().__init__()
        self.layers = nn.Sequential()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, layers[0]))
        self.layers.append(nn.ReLU())
        if input_dropout > 0:
            self.layers.append(nn.Dropout(input_dropout))
        
        # Hidden layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
            if i < len(layers) - 1:  # no ReLU/Dropout after last layer
                self.layers.append(nn.ReLU())
                if hidden_dropout > 0:
                    self.layers.append(nn.Dropout(hidden_dropout))
    
    def forward(self, x):
        return self.layers(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, learning_rate=1e-3, early_stopping=True, patience=5, min_delta=0.01):
        """
        LightningModule wrapping an encoder (or autoencoder).
        Provides training/validation/test steps and optimizer setup.
        """
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.encoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """Initialize Linear layers with Kaiming-normal (good for ReLU)."""
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)

    def training_step(self, batch, batch_idx):
        """One training step: compute MSE loss and log it."""
        x, y = batch
        z = self.encoder(x)
        loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step: compute MSE loss on validation set."""
        x, y = batch
        z = self.encoder(x)
        val_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("val_loss", val_loss)
        return val_loss
        
    def test_step(self, batch, batch_idx):
        """Test step: compute MSE loss on test set."""
        x, y = batch
        z = self.encoder(x)
        test_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        """Forward pass through encoder."""
        return self.encoder(x)

    def configure_optimizers(self):
        """Use SGD optimizer with momentum."""
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.5)

    def configure_callbacks(self):
        """Add EarlyStopping callback if enabled."""
        if not self.early_stopping:
            return []
        return [EarlyStopping(
            monitor="val_loss",
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=True,
            mode="min"
        )]
