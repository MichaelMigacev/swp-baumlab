import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
import pickle
import gzip
from itertools import product
import os
from datetime import datetime
from model import Encoder, LitAutoEncoder
import json
import math  # für isnan
import random
import numpy as np


# 1. Load data
def load_data(data_file):
    with gzip.open(data_file, mode='rb') as f:
        return pickle.load(f)
    
# 2. Data Module
class SynergyDataModule(L.LightningDataModule):
    def __init__(self, data_file, batch_size=64):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        
    def prepare_data(self):
        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names, filtered_feature_origin = load_data(self.data_file)
        
        self.train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        self.val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        self.test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        self.index_names = index_names
        self.input_size = X_train.shape[1]
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# 3. Manual Grid Search mit Verbesserungen
# CHANGE HERE
def run_manual_grid_search(data_file="test0val1normtanh_norm.p.gz", seed=42):
    # Seed setzen für Reproduzierbarkeit
    # Python & NumPy & PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Für deterministische CUDA-Berechnung (langsamer, aber reproduzierbar)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize data
    datamodule = SynergyDataModule(data_file)
    datamodule.prepare_data()
    
    # Define search space
    search_space = {
        "hidden_layers": [
        [8192, 8192, 1], [4096, 4096, 1], [2048, 2048, 1],
        [8192, 4096, 1], [4096, 2048, 1], [4096, 4096, 4096, 1],
        [2048, 2048, 2048, 1], [4096, 2048, 1024, 1],
        [8192, 4096, 2048, 1]
        ],
        "learning_rate": [0.01, 0.001, 0.0001, 1e-5],
        "dropout": [
            {"input": 0.0, "hidden": 0.0},
            {"input": 0.2, "hidden": 0.5}
        ]
    }

    # Create results directory
    os.makedirs("grid_search_results", exist_ok=True)
    #CHANGE HERE
    results_file = f"grid_search_results/resultstest0val1normtanh_norm.csv"
    
    # Write header
    with open(results_file, "w") as f:
        f.write("hidden_layers,learning_rate,dropout_input,dropout_hidden,val_loss,test_loss\n")
    
    best_val_loss = float('inf')
    best_config = None
    
    # Generate all combinations
    for config in product(*search_space.values()):
        layers, lr, dropout = config
        early_stop = True
        
        print(f"\nTraining config: layers={layers}, lr={lr:.0e}, dropout={dropout}")
        
        # Build model
        encoder = Encoder(
            input_size=datamodule.input_size,
            layers=layers,
            input_dropout=dropout["input"],
            hidden_dropout=dropout["hidden"]
        )
        
        model = LitAutoEncoder(
            encoder=encoder,
            learning_rate=lr,
            early_stopping=early_stop,
            patience=5,
            min_delta=2
        )
        
        # Trainer setup
        trainer = L.Trainer(
            max_epochs=40,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            callbacks=[
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min"
                )
            ]
        )
        
        try:
            # Train
            trainer.fit(model, datamodule=datamodule)
            
            # Validate
            val_result = trainer.validate(model, datamodule=datamodule)
            val_loss = val_result[0]["val_loss"]
            
            # Test
            test_result = trainer.test(model, datamodule=datamodule)
            test_loss = test_result[0]["test_loss"]
            
            # Prüfe ob NaN in Loss
            if math.isnan(val_loss) or math.isnan(test_loss):
                print(f"Warnung: NaN Loss bei config {layers}, lr={lr}")
                continue  # skip this run
            
        except Exception as e:
            print(f"Fehler bei config {layers}, lr={lr}: {e}")
            continue  # Fehler überspringen, weiter mit nächster Config
        
        # Save results
        with open(results_file, "a") as f:
            f.write(f"{str(layers)},{lr},{dropout['input']},{dropout['hidden']},{val_loss},{test_loss}\n")
        
        # Track best config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = {
                "layers": layers,
                "lr": lr,
                "dropout": dropout,
                "val_loss": val_loss,
                "test_loss": test_loss
            }
            print(f"New best config! Val Loss: {val_loss:.4f}")
    
    print("\nBest configuration found:")
    print(best_config)
    
    # Save best config
    with open(f"grid_search_results/best_config.json", "w") as f:
        json.dump(best_config, f)
    
    return best_config

if __name__ == "__main__":
    run_manual_grid_search()
