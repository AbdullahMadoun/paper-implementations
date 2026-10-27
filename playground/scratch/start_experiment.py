import subprocess
import os

script_content = """# -*- coding: utf-8 -*-
from datasets import load_dataset
import numpy as np
import pandas as pd
import os

print("Loading dataset (14.69 GB)...")
ds = load_dataset("UniverseTBD/legacysurvey_hsc_embeddings", split="train")
print(f"Total rows: {len(ds)}")
print("Features:", list(ds.features.keys()))

def MKNN(d1, d2, k):
    # Filter out rows where either embedding is None
    valid_idx = [i for i, (x, y) in enumerate(zip(d1, d2)) if x is not None and y is not None]
    if not valid_idx:
        return 0.0
    
    d1_clean = np.array([d1[i] for i in valid_idx], dtype=np.float32)
    d2_clean = np.array([d2[i] for i in valid_idx], dtype=np.float32)
    
    # Truncate outliers above 95th percentile
    d1_clean = np.clip(d1_clean, -np.percentile(np.abs(d1_clean), 95), np.percentile(np.abs(d1_clean), 95))
    d2_clean = np.clip(d2_clean, -np.percentile(np.abs(d2_clean), 95), np.percentile(np.abs(d2_clean), 95))
    
    # Normalize for cosine similarity
    d1_norm = d1_clean / np.linalg.norm(d1_clean, axis=1, keepdims=True)
    d2_norm = d2_clean / np.linalg.norm(d2_clean, axis=1, keepdims=True)
    
    # Batched KNN index calculation using BLAS matrix multiplication
    def get_knn(t, k, batch_size=5000):
        n = t.shape[0]
        knn_indices = []
        for i in range(0, n, batch_size):
            batch = t[i:i+batch_size]
            # dot product (B x N)
            sim = np.dot(batch, t.T)
            # Find top-k indices (largest similarity corresponds to smallest distance)
            indices = np.argpartition(-sim, k, axis=1)[:, :k]
            # Sort the indices within each row to ensure consistency
            for row_idx in range(len(indices)):
                row = indices[row_idx]
                indices[row_idx] = row[np.argsort(-sim[row_idx, row])]
            knn_indices.append(indices)
        return np.concatenate(knn_indices, axis=0)

    knn_1 = get_knn(d1_norm, k)
    knn_2 = get_knn(d2_norm, k)
    
    # Optimize intersection check using Python sets
    matches = [len(set(knn_1[i]) & set(knn_2[i])) for i in range(knn_1.shape[0])]
    return np.sum(matches) / (float(knn_1.shape[0]) * k)

names = list(ds.features.keys())[1:]
names_legacy = [n for n in names if n.endswith("legacysurvey")]
names_hsc = [n for n in names if n.endswith("hsc")]

print(f"HSC columns ({len(names_hsc)}): {names_hsc}")
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
print("=== Final MKNN % Table ===")
print(df.to_string())

df.to_csv("/content/mknn_results.csv")
print("Saved to /content/mknn_results.csv")
"""

with open("/content/colab_mknn.py", "w") as f:
    f.write(script_content)

print("Starting background process...")
# Run the python script in background using subprocess
p = subprocess.Popen(
    ["python", "-u", "/content/colab_mknn.py"],
    stdout=open("/content/colab_mknn.log", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp # decouple from parent process group
)
print(f"Started! PID: {p.pid}")
