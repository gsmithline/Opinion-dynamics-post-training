import pickle
import pandas as pd

with open("pokec_dataset/lcc_profiles_relation_to_smoking.pk", "rb") as f:
    df = pickle.load(f)

print("columns:", list(df.columns))
print("shape:", df.shape)
for col in ["age", "gender", "relation_to_alcohol", "relation_to_smoking"]:
    if col in df.columns:
        vc = df[col].value_counts(dropna=False).head(10)
        print(f"\n--- {col} (dtype={df[col].dtype}) ---")
        print(vc)
        print("n_nan:", df[col].isna().sum(), "n_empty_str:", (df[col].astype(str) == "").sum())

print("\nFirst 5 rows (relevant cols):")
cols = [c for c in ["age","gender","relation_to_alcohol","relation_to_smoking"] if c in df.columns]
print(df[cols].head().to_string())
