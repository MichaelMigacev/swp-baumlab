import torch
import os
import lightning.pytorch as L
from model import Encoder, LitAutoEncoder
from torch.utils.data import TensorDataset, DataLoader
from grid_search import SynergyDataModule
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import random
import numpy as np


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_model(
    seed=23,
    data_file="test0val1normtanh_norm.p.gz",
    layers=[8192, 4096, 1],
    learning_rate=0.00001,
    dropout={"input": 0.2, "hidden": 0.5},
    save_path=None
):
    set_all_seeds(seed)

    # Derive save path if not provided
    if save_path is None:
        base_name = os.path.splitext(data_file)[0]
        save_path = f"saved_models/{base_name}_model.pt"

    # Prepare data
    datamodule = SynergyDataModule(data_file)
    datamodule.prepare_data()

    # Create model
    encoder = Encoder(
        input_size=datamodule.input_size,
        layers=layers,
        input_dropout=dropout["input"],
        hidden_dropout=dropout["hidden"]
    )

    model = LitAutoEncoder(
        encoder=encoder,
        learning_rate=learning_rate,
        early_stopping=True,
        patience=10,
        min_delta=0.5
    )

    # Set up CSV logger
    logger = CSVLogger("logs", name="synergy_model")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=200,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Test model
    test_result = trainer.test(model, datamodule=datamodule)
    print(f"Test loss: {test_result[0]['test_loss']:.4f}")

    # Save model weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_single_model()
