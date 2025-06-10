import os
import gzip
import pickle
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from captum.attr import FeatureAblation
from grid_search import SynergyDataModule
from model import Encoder, load_model
from torch.utils.data import DataLoader, TensorDataset

# ----------------- Settings -----------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

input_size = 8846
layers = [8192, 4096, 1]
input_dropout = 0.2
hidden_dropout = 0.5
model_path = "/home/mi/michaef01/Dokumente/PSem/saved_models/test0val1normtanh_norm.p_model.pt"
data_file = "/home/mi/michaef01/Dokumente/PSem/test0val1normtanh_norm.p.gz"
output_dir = "ablation_outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load Model -----------------

encoder = load_model(
    Encoder,
    path=model_path,
    input_size=input_size,
    layers=layers,
    input_dropout=input_dropout,
    hidden_dropout=hidden_dropout
)
encoder.to(device)
encoder.eval()
print("Model loaded from:", model_path)

# ----------------- Load Data -----------------

with gzip.open(data_file, 'rb') as f:
    data = pickle.load(f)

X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names = data[:9]

try:
    filtered_feature_origin = data[9]
    feature_names = list(filtered_feature_origin)
    print("Loaded filtered feature names.")
except IndexError:
    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    print("Filtered feature names not found; using default indices.")

# ----------------- Prepare Test Data -----------------

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
sample_inputs, _ = next(iter(test_loader))
sample_inputs = sample_inputs.to(device)

# ----------------- Captum Feature Ablation -----------------

ablation = FeatureAblation(encoder)

# Target: explain first output neuron (for example)
target_output = 0

# Compute attributions
print("Computing Feature Ablation attributions...")
ablation_attrs = ablation.attribute(sample_inputs[:40], target=target_output)  # (samples, features)
ablation_attrs = ablation_attrs.detach().cpu().numpy()

# ----------------- Ablation Summary Plot -----------------

summary_plot_path = os.path.join(output_dir, "ablation_summary.png")
mean_abs_ablation = np.abs(ablation_attrs).mean(axis=0)

plt.figure(figsize=(14, 4))
plt.bar(range(len(mean_abs_ablation)), mean_abs_ablation)
plt.xlabel("Feature Index")
plt.ylabel("Mean |Ablation Attribution|")
plt.title("Feature Ablation Mean Importance")
plt.tight_layout()
plt.savefig(summary_plot_path)
plt.clf()
print(f"Saved Feature Ablation summary plot to: {summary_plot_path}")

# ----------------- Ablation Heatmap (Top N Features) -----------------

N = 50
top_indices = np.argsort(mean_abs_ablation)[-N:]
heatmap_data = ablation_attrs[:, top_indices]
heatmap_feature_names = [feature_names[i] for i in top_indices]
sample_labels = index_names[:heatmap_data.shape[0]]

plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_data,
    xticklabels=heatmap_feature_names,
    yticklabels=sample_labels,
    cmap="coolwarm",
    center=0,
    cbar_kws={'label': 'Ablation Value'}
)
plt.title(f"Feature Ablation Heatmap (Top {N} Features)")
plt.xlabel("Feature")
plt.ylabel("Sample")
heatmap_path = os.path.join(output_dir, "ablation_heatmap.png")
plt.tight_layout()
plt.savefig(heatmap_path)
plt.clf()
print(f"Saved ablation heatmap to: {heatmap_path}")
