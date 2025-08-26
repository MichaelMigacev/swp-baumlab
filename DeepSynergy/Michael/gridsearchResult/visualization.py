import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# Show all rows in pandas DataFrames
pd.set_option('display.max_rows', None)

# === === === USER SETTINGS === === ===
# List of CSV files to load
csv_files = [
    "grid_search_results/resultstest0val1normnorm.csv",
    "grid_search_results/resultstest0val1normtanh_norm.csv",
    "grid_search_results/resultstest0val1normtanh.csv"
]

# Function to load custom CSV files with specific format
def load_custom_csv(file_path, version):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Skip header line
            if "hidden_layers" in line:
                continue
            split_idx = line.find(']') + 1
            if split_idx == 0:
                continue
            layer_str = line[:split_idx]
            rest = line[split_idx+1:].strip().split(',')
            row = [layer_str] + rest
            # Skip lines that do not have exactly 6 elements
            if len(row) != 6:
                print(f"⚠️ Line skipped: {line.strip()}")
                continue
            data.append(row)

    # Convert list of rows to pandas DataFrame
    df = pd.DataFrame(data, columns=[
        "hidden_layers", "learning_rate", "dropout_input", "dropout_hidden", "val_loss", "test_loss"
    ])
    # Convert hidden_layers from string to tuple (for grouping purposes)
    df["hidden_layers"] = df["hidden_layers"].apply(lambda x: tuple(ast.literal_eval(x)))
    # Convert columns to float
    df["learning_rate"] = df["learning_rate"].astype(float)
    df["dropout_input"] = df["dropout_input"].astype(float)
    df["dropout_hidden"] = df["dropout_hidden"].astype(float)
    df["val_loss"] = df["val_loss"].astype(float)
    df["test_loss"] = df["test_loss"].astype(float)
    # Add version column
    df["version"] = version
    return df

# Load all CSV files into a list of DataFrames
dfs = []
for file in csv_files:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue
    version_name = os.path.splitext(os.path.basename(file))[0]
    df = load_custom_csv(file, version_name)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
full_df = pd.concat(dfs, ignore_index=True)

# Group by configuration and version, compute mean validation/test loss
grouped_df = (
    full_df
    .groupby(["hidden_layers", "learning_rate", "dropout_input", "dropout_hidden", "version"], as_index=False)
    .agg({
        "val_loss": "mean",
        "test_loss": "mean"
    })
)

# Create a string representation of the configuration for plotting
grouped_df["config"] = (
    grouped_df["hidden_layers"].astype(str)
    + "_lr=" + grouped_df["learning_rate"].astype(str)
    + "_drop=" + grouped_df["dropout_input"].astype(str) + "-" + grouped_df["dropout_hidden"].astype(str)
)
# Version used for sorting
sort_version = "resultstest0val1normtanh_norm"

# Get order of configurations sorted by test_loss for the selected version
sort_order = (
    grouped_df[grouped_df["version"] == sort_version]
    .sort_values("test_loss")["config"]
    .unique()
)

# === Plot 1: Validation Loss per configuration and version ===
plt.figure(figsize=(14, 6))
sns.barplot(data=grouped_df, x="config", y="val_loss", hue="version", order=sort_order)
plt.xticks(rotation=90)
plt.title("Validation Loss comparison per configuration and version (averaged)")
plt.tight_layout()
plt.savefig("val_loss_comparison_by_version.png")
plt.show()

# === Plot 2: Test Loss per configuration and version ===
plt.figure(figsize=(14, 6))
sns.barplot(data=grouped_df, x="config", y="test_loss", hue="version", order=sort_order)
plt.xticks(rotation=90)
plt.title("Test Loss comparison per configuration and version (averaged)")
plt.ylim(200, grouped_df["test_loss"].max() * 1.05)  # y-axis starts at 200 with a little padding
plt.tight_layout()
plt.savefig("test_loss_comparison_by_version.png")
plt.show()
