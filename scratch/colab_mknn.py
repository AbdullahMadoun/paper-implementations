from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

print("Loading dataset (14.69 GB)...")
ds = load_dataset("UniverseTBD/legacysurvey_hsc_embeddings", split="train")
print(f"Total rows: {len(ds)}")
print("Features:", list(ds.features.keys()))

def MKNN(d1, d2, k):
    # Filter out rows where either embedding is None
    valid_idx = [i for i, (x, y) in enumerate(zip(d1, d2)) if x is not None and y is not None]
    if not valid_idx:
        return 0.0
    d1_clean = np.array([d1[i] for i in valid_idx])
    d2_clean = np.array([d2[i] for i in valid_idx])
    # Truncate outliers above 95th percentile
    d1_clean = np.clip(d1_clean, -np.percentile(np.abs(d1_clean), 95), np.percentile(np.abs(d1_clean), 95))
    d2_clean = np.clip(d2_clean, -np.percentile(np.abs(d2_clean), 95), np.percentile(np.abs(d2_clean), 95))
    knn_1 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(d1_clean)
    knn_2 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(d2_clean)
    knn_1 = knn_1.kneighbors(return_distance=False)
    knn_2 = knn_2.kneighbors(return_distance=False)
    matches = [len(np.intersect1d(knn_1[i], knn_2[i])) for i in range(knn_1.shape[0])]
    return np.sum(matches) / (float(knn_1.shape[0]) * k)

names = list(ds.features.keys())[1:]
names_legacy = [n for n in names if n.endswith("legacysurvey")]
names_hsc = [n for n in names if n.endswith("hsc")]

print(f"\nHSC columns ({len(names_hsc)}): {names_hsc}")
print(f"Legacy columns ({len(names_legacy)}): {names_legacy}")

# Compute MKNN for HSC pairs
hsc_pairs = {}
for i in range(1, len(names_hsc)):
    if names_hsc[i].split("_")[0] == names_hsc[i-1].split("_")[0]:
        key = f"{names_hsc[i-1]} vs {names_hsc[i]}"
        print(f"Computing: {key} ...")
        hsc_pairs[key] = MKNN(ds[names_hsc[i-1]], ds[names_hsc[i]], 10)
        print(f"  -> {hsc_pairs[key]:.4f}")

# Compute MKNN for Legacy Survey pairs
legacy_pairs = {}
for i in range(1, len(names_legacy)):
    if names_legacy[i].split("_")[0] == names_legacy[i-1].split("_")[0]:
        key = f"{names_legacy[i-1]} vs {names_legacy[i]}"
        print(f"Computing: {key} ...")
        legacy_pairs[key] = MKNN(ds[names_legacy[i-1]], ds[names_legacy[i]], 10)
        print(f"  -> {legacy_pairs[key]:.4f}")

# Build results table
data = {}
for k1, v1 in hsc_pairs.items():
    model_key = k1.replace("_hsc", "")
    data[model_key] = {"hsc": round(v1 * 100, 2), "legacysurvey": None}
for k2, v2 in legacy_pairs.items():
    model_key = k2.replace("_legacysurvey", "")
    if model_key in data:
        data[model_key]["legacysurvey"] = round(v2 * 100, 2)
    else:
        data[model_key] = {"hsc": None, "legacysurvey": round(v2 * 100, 2)}

df = pd.DataFrame.from_dict(data, orient="index")
print("\n=== Final MKNN % Table ===")
print(df.to_string())

# Save to CSV
df.to_csv("mknn_results.csv")
print("\nSaved to mknn_results.csv")
