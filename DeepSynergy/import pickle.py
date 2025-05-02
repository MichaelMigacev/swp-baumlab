import pickle

# Pfad zur Datei
file_path = "data/data_test_fold0_tanh.p"

# Lade Pickle-Datei
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Überblick
print("📦 Schlüssel im Datensatz:", data.keys())

x = data["x"]
y = data["y"]

print("🧪 Eingabedaten (x):", x.shape)
print("🎯 Zielwerte (y):", y.shape)
