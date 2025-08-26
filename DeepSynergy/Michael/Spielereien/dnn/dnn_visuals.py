import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# === Schritt 1: Datei laden ===
with open("results_dnn/test_results.pkl", "rb") as f:
    obj = pickle.load(f)

# === Schritt 2: Predictions & Targets extrahieren ===
preds = obj['predictions']
targets = obj['targets']

# === Schritt 3: Fehlermaße berechnen ===
mse = mean_squared_error(targets, preds)
mae = mean_absolute_error(targets, preds)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# === Schritt 4: Scatterplot zeichnen ===
plt.figure(figsize=(8, 6))
plt.scatter(targets, preds, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predictions vs. True Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_predictions_vs_targets.png")
plt.show()
