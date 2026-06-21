from datasets import load_dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Load dataset
ds = load_dataset("UniverseTBD/jwst_hsc_embeddings", split="train")
ds = ds.filter(lambda row: row['dino_base_jwst'] is not None)

def MKNN(d1, d2, k=10):
    d1 = np.array(d1)
    d2 = np.array(d2)
    knn_1 = NearestNeighbors(n_neighbors=k, metric="cosine").fit(d1)
    knn_2 = NearestNeighbors(n_neighbors=k, metric="cosine").fit(d2)
    knn_1 = knn_1.kneighbors(return_distance=False)
    knn_2 = knn_2.kneighbors(return_distance=False)
    overlap = [len(set(a).intersection(b)) for a, b in zip(knn_1, knn_2)]
    return np.mean(overlap) / k

names = list(ds.features.keys())[1:]
names_hsc = [n for n in names if n.endswith("_hsc")]

cross_modal_pairs = {}
for col_hsc in names_hsc:
    col_jwst = col_hsc.replace("_hsc", "_jwst")
    if col_jwst in ds.features:
        key = col_hsc.replace("_hsc", "") # e.g. "dino_base"
        print(f"Computing cross-modal MKNN for: {key} ...")
        score = MKNN(ds[col_hsc], ds[col_jwst], 10)
        cross_modal_pairs[key] = score
        print(f"  -> {score:.4f}")

# Group by model family
print("\nResults:")
for k, v in cross_modal_pairs.items():
    print(f"{k}: {v*100:.2f}%")
