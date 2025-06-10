import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# Optional: Zeige mehr Zeilen
pd.set_option('display.max_rows', None)

# === === === USER SETTINGS === === ===
csv_files = [
    "grid_search_results/resultstest0val1normtanh_norm_20250525_170203.csv",
    "grid_search_results/resultstest0val1normtanh_norm_20250525_170256.csv",
    "grid_search_results/resultstest0val1normtanh_norm_20250525_170257.csv",
    "grid_search_results/resultstest0val1normtanh_norm_20250525_170259.csv"
]
labels = ["Run 1", "Run 2", "Run 3", "Run 4"]  # Muss zur Anzahl csv_files passen
# === === === === === === === === === ==

assert len(csv_files) == len(labels), "csv_files und labels müssen gleich lang sein."

def load_custom_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if "hidden_layers" in line:
                continue  # Header überspringen
            split_idx = line.find(']') + 1
            if split_idx == 0:
                continue
            layer_str = line[:split_idx]
            rest = line[split_idx+1:].strip().split(',')
            row = [layer_str] + rest
            if len(row) != 6:
                print(f"⚠️ Zeile übersprungen: {line.strip()}")
                continue
            data.append(row)

    df = pd.DataFrame(data, columns=[
        "hidden_layers", "learning_rate", "dropout_input", "dropout_hidden", "val_loss", "test_loss"
    ])
    df["hidden_layers"] = df["hidden_layers"].apply(ast.literal_eval)
    df["learning_rate"] = df["learning_rate"].astype(float)
    df["dropout_input"] = df["dropout_input"].astype(float)
    df["dropout_hidden"] = df["dropout_hidden"].astype(float)
    df["val_loss"] = df["val_loss"].astype(float)
    df["test_loss"] = df["test_loss"].astype(float)
    return df

# Alle CSV-Dateien laden
dfs = []
for file, label in zip(csv_files, labels):
    if not os.path.exists(file):
        print(f"❌ Datei nicht gefunden: {file}")
        continue
    df = load_custom_csv(file)
    df["run"] = label
    dfs.append(df)

# Daten kombinieren
full_df = pd.concat(dfs, ignore_index=True)

# Konfiguration für Gruppierung als String
full_df["config"] = (
    full_df["hidden_layers"].astype(str)
    + "_lr=" + full_df["learning_rate"].astype(str)
    + "_drop=" + full_df["dropout_input"].astype(str) + "-" + full_df["dropout_hidden"].astype(str)
)

# === Plot 1: Val Loss über alle Runs ===
plt.figure(figsize=(14, 6))
sns.barplot(data=full_df, x="config", y="val_loss", hue="run")
plt.xticks(rotation=90)
plt.title("Validation Loss Vergleich über Konfigurationen")
plt.tight_layout()
plt.savefig("val_loss_comparison_tanhnorm.png")
plt.show()

# === Plot 2: Test Loss über alle Runs ===
plt.figure(figsize=(14, 6))
sns.barplot(data=full_df, x="config", y="test_loss", hue="run")
plt.xticks(rotation=90)
plt.title("Test Loss Vergleich über Konfigurationen")
plt.tight_layout()
plt.savefig("test_loss_comparison_tanhnorm.png")
plt.show()

# === Optional: Streuung anzeigen (Boxplot) ===
plt.figure(figsize=(14, 6))
sns.boxplot(data=full_df, x="config", y="val_loss")
plt.xticks(rotation=90)
plt.title("Streuung des Validation Loss pro Konfiguration")
plt.tight_layout()
plt.savefig("val_loss_boxplot_tanhnorm.png")
plt.show()
