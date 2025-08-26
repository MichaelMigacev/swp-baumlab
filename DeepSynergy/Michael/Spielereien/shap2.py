import os
import gzip
import pickle
import torch
import torch.nn as nn
import shap
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

from grid_search import SynergyDataModule
from model import Encoder, load_model

# ------------------ Settings ------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ORIG_INPUT_SIZE = 8846
LAYERS = [8192, 4096, 1]
INPUT_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.5

model_path = "saved_models/test0val1normtanh_norm.p_model.pt"
data_file = "test0val1normtanh_norm.p.gz"
output_dir = "shap_outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load Data ------------------
print("Loading...")
with gzip.open(data_file, 'rb') as f:
    data = pickle.load(f)

# Unpack the correct number of variables
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names, f_features_origin, f_feature_group = data

# ------------------ No Feature Filtering ------------------

X_train_filt = X_train
X_test_filt = X_test

# Feature names:
feature_names = list(f_features_origin)

# ------------------ Load Model ------------------

encoder = Encoder(
    input_size=ORIG_INPUT_SIZE,
    layers=LAYERS,
    input_dropout=INPUT_DROPOUT,
    hidden_dropout=HIDDEN_DROPOUT
)

state = torch.load(model_path, map_location=device)
encoder.load_state_dict(state)
encoder.to(device)
encoder.eval()

# ------------------ Prepare DataLoader ------------------

test_ds = TensorDataset(
    torch.tensor(X_test_filt, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
sample_X, _ = next(iter(test_loader))
sample_X = sample_X.to(device)

# ------------------ SHAP Explainer ------------------

print("Explaining...")

try:
    bg_np = shap.utils.sample(sample_X.cpu().numpy(), 8000)
except AttributeError:
    bg_np = shap.sample(sample_X.cpu().numpy(), 8000)

bg_t = torch.tensor(bg_np, dtype=torch.float32).to(device)
explainer = shap.DeepExplainer(encoder, bg_t)

test_np = shap.utils.sample(sample_X.cpu().numpy(), 40)
test_t = torch.tensor(test_np, dtype=torch.float32).to(device)

sv = explainer.shap_values(test_t)
sv = sv[0] if isinstance(sv, list) else sv
sv = np.squeeze(sv)

# ------------------ SHAP Heatmap ------------------

print("Generating heatmap...")

mean_abs_shap = np.abs(sv).mean(axis=0)
top_k = 40
top_k_idx = np.argsort(mean_abs_shap)[::-1][:top_k]

sv_topk = sv[:, top_k_idx]
feature_names_topk = [feature_names[i] for i in top_k_idx]

sv_df_topk = pd.DataFrame(sv_topk, columns=feature_names_topk)

plt.figure(figsize=(15, 10))
sns.heatmap(
    sv_df_topk.T,
    cmap="coolwarm",
    center=0,
    cbar_kws={'label': 'SHAP value'},
    xticklabels=False,
    yticklabels=True
)
plt.title(f"SHAP Value Heatmap (Top {top_k} features x samples)")
plt.ylabel("Features")
plt.xlabel("Samples")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_heatmap_topk.png"))
plt.clf()

# ------------------ SHAP Summary Plot ------------------

plt.figure()
shap.summary_plot(
    sv,
    test_np,
    feature_names=feature_names,
    max_display=50,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary.png"))
plt.clf()

# ------------------ SHAP Grouped Bar Plot ------------------

mean_abs = np.abs(sv).mean(axis=0)
grouped = defaultdict(list)

# Gruppieren nach f_feature_group (statt filtered_features_origin)
for idx in range(len(f_feature_group)):
    grp = f_feature_group[idx]
    grouped[grp].append(idx)

group_means = {
    grp: mean_abs[indices].mean() for grp, indices in grouped.items()
}

top10 = sorted(group_means.items(), key=lambda x: x[1], reverse=True)[:10]
grps, scores = zip(*top10)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(scores), y=list(grps), palette="viridis")
plt.title("Mean |SHAP| by Feature Group (Top 10)")
plt.xlabel("Mean |SHAP|")
plt.ylabel("Feature Group")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_grouped_bar.png"))
plt.clf()

print(f"✅ Saved all SHAP plots to: `{output_dir}`")
