import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Lade die Daten
with open("data/data_test_fold0_tanh.p", "rb") as f:
    data = pickle.load(f)

# Extrahiere Eingaben und Ziele
x_train, x_val, x_test = data[0], data[1], data[2]
y_train, y_val, y_test = data[4], data[5], data[6]

# In Torch-Tensoren umwandeln
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Dataset + DataLoader
train_dataset = TensorDataset(x_train, y_train)
val_dataset   = TensorDataset(x_val, y_val)
test_dataset  = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)

print("✅ PyTorch Datasets geladen:")
print(f"Train: {len(train_dataset)} Beispiele")
print(f"Val:   {len(val_dataset)} Beispiele")
print(f"Test:  {len(test_dataset)} Beispiele")
