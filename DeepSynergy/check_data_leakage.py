import pandas as pd

# 1. Datei laden
df = pd.read_csv("data/drug_combination_data.txt", sep="\t")

# 2. AB/BA-neutral darstellen
df["combo_set"] = df.apply(lambda row: frozenset([row["Drug1_ID"], row["Drug2_ID"]]), axis=1)

# 3. Nach Set (Train/Test) trennen
train_df = df[df["Set"] == "Train"]
test_df = df[df["Set"] == "Test"]

train_combos = set(train_df["combo_set"])
test_combos = set(test_df["combo_set"])

# 4. Schnittmenge prüfen (AB in Train, BA in Test = gleiche Kombination)
overlap = train_combos & test_combos

print("✅ Anzahl eindeutiger Drogenkombinationen (reihenfolgeunabhängig):")
print(f"  Train: {len(train_combos)}")
print(f"  Test : {len(test_combos)}")
print()
if overlap:
    print(f"⚠️  Data Leakage gefunden! {len(overlap)} Drogenkombinationen in beiden Sets.")
    print("👉 Beispiele:")
    for i, combo in enumerate(list(overlap)[:5]):
        print(f"   {sorted(list(combo))}")
else:
    print("✅ Kein Data Leakage durch AB/BA-Kombis gefunden.")
