import torch
import torch.nn as nn
import torch.optim as optim
from model import DeepSynergyModel
from dataset_loader import train_loader, val_loader

# 🔧 Modellparameter
input_size = 8846
learning_rate = 1e-4
num_epochs = 3  # Weniger zum schnellen Testen

# 📦 Modell, Loss, Optimizer
model = DeepSynergyModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    print("🚀 Starte Training")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # 🔍 Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_outputs = model(val_x)
                val_loss += criterion(val_outputs, val_y).item() * val_x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"📈 Epoche {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

# 📦 Modell speichern
torch.save(model.state_dict(), "deepsynergy_model.pt")
print("💾 Modell gespeichert als 'deepsynergy_model.pt'")