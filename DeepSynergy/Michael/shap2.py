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

from grid_search import SynergyDataModule
from model import Encoder, load_model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import VarianceThreshold
from collections import defaultdict

# ------------------ Settings ------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ORIG_INPUT_SIZE = 8846   # what the saved model was trained on
LAYERS = [8192, 4096, 1]
INPUT_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.5

model_path  = "/home/mi/michaef01/Dokumente/PSem/saved_models/test0val1normtanh_norm.p_model.pt"
data_file   = "/home/mi/michaef01/Dokumente/PSem/test0val1normtanh_norm.p.gz"
output_dir  = "shap_outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load Data ------------------

with gzip.open(data_file, 'rb') as f:
    data = pickle.load(f)

# Unpack (adjust if you had different splits)
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test, index_names = data[:9]

# Recover original feature names
try:
    filtered_feature_origin = data[9]
    feature_names = list(filtered_feature_origin)
except IndexError:
    feature_names = [f"feature_{i}" for i in range(ORIG_INPUT_SIZE)]

# ------------------ Fit VarianceThreshold (train-only) ------------------

print("Fitting VarianceThreshold on training data…")
var_thresh = VarianceThreshold(threshold=1e-5)
X_train_filt = var_thresh.fit_transform(X_train)    # fit on train
X_test_filt  = var_thresh.transform(X_test)         # apply to test

selected_indices = var_thresh.get_support(indices=True)
feature_names    = [feature_names[i] for i in selected_indices]
print(f"  → Kept {len(selected_indices)} / {ORIG_INPUT_SIZE} features")

# ------------------ Load and Prune Model ------------------

# 1. instantiate with the original input_size
encoder = Encoder(
    input_size=ORIG_INPUT_SIZE,
    layers=LAYERS,
    input_dropout=INPUT_DROPOUT,
    hidden_dropout=HIDDEN_DROPOUT
)
# 2. load its full state dict
state = torch.load(model_path, map_location=device)
encoder.load_state_dict(state)
encoder.to(device)
encoder.eval()

# 3. prune the first Linear
old_linear = encoder.layers[0]  # assume Encoder defines self.layers as a ModuleList
W_old = old_linear.weight.data          # shape [8192, 8846]
b_old = old_linear.bias.data            # shape [8192]

# build a new Linear that accepts only the filtered inputs
in_feats  = len(selected_indices)       # e.g. 8838
out_feats = old_linear.out_features     # 8192

new_linear = nn.Linear(in_feats, out_feats, bias=True)
# copy over the pruned weights & biases
new_linear.weight.data = W_old[:, selected_indices].clone()
new_linear.bias.data   = b_old.clone()

# replace in the model
encoder.layers[0] = new_linear

print("Model’s first layer pruned to match filtered features.")
print(f"  → New input dimension: {encoder.layers[0].in_features}")

# ------------------ Prepare DataLoader ------------------

test_dataset = TensorDataset(
    torch.tensor(X_test_filt, dtype=torch.float32),
    torch.tensor(y_test,    dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
sample_inputs, _ = next(iter(test_loader))
sample_inputs = sample_inputs.to(device)

# ------------------ SHAP Explainer ------------------

# use 100 background samples
background_np = shap.sample(sample_inputs.cpu().numpy(), 100)
background_t  = torch.tensor(background_np, dtype=torch.float32).to(device)

explainer = shap.DeepExplainer(encoder, background_t)

# sample 40 test points for explanation
test_np   = shap.sample(sample_inputs.cpu().numpy(), 40)
test_t    = torch.tensor(test_np, dtype=torch.float32).to(device)

shap_values = explainer.shap_values(test_t)
shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
shap_values = np.squeeze(shap_values)

# ------------------ SHAP Summary Plot ------------------

summary_path = os.path.join(output_dir, "shap_summary.png")
shap.summary_plot(
    shap_values,
    test_np,
    feature_names=feature_names,
    max_display=50,
    show=False
)
plt.tight_layout()
plt.savefig(summary_path)
plt.clf()

# ------------------ SHAP Grouped Bar Plot ------------------

def group_feature_indices(names):
    groups = defaultdict(list)
    for idx, name in enumerate(names):
        prefix = name.split("_")[0] if "_" in name else "misc"
        groups[prefix].append(idx)
    return groups

grouped_indices = group_feature_indices(feature_names)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

group_means = {
    grp: mean_abs_shap[idxs].mean()
    for grp, idxs in grouped_indices.items()
}

# top‐10 groups
topn = sorted(group_means.items(), key=lambda x: x[1], reverse=True)[:10]
grps, scores = zip(*topn)

plt.figure(figsize=(10, 5))
sns.barplot(x=scores, y=grps, palette="viridis")
plt.title("Mean |SHAP| by Feature Group (Top 10)")
plt.xlabel("Mean |SHAP|")
plt.ylabel("Feature Group")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_grouped_bar.png"))
plt.clf()

print(f"✅ Saved SHAP plots to '{output_dir}'")
