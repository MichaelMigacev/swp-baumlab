import pickle

file_path = "data/data_test_fold0_tanh.p"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print("📦 Geladener Typ:", type(data))

# Wenn es ein Tuple ist, zeige seine Länge und Typen
if isinstance(data, tuple):
    print("📦 Tuple mit", len(data), "Elementen:")
    for i, item in enumerate(data):
        print(f"  Element {i}: Typ = {type(item)}, Shape (falls ndarray): {getattr(item, 'shape', 'kein shape')}")

# Wenn es ein Dict ist (zur Sicherheit)
elif isinstance(data, dict):
    print("📦 Dict mit Schlüsseln:", data.keys())
