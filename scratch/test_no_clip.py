from datasets import load_dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load dataset
ds = load_dataset("UniverseTBD/jwst_hsc_embeddings", split="train")

# Filter out None rows
ds = ds.filter(lambda row: row['dino_base_jwst'] is not None)

def MKNN_no_clip(d1, d2, k=10):
    d1 = np.array(d1)
    d2 = np.array(d2)
    knn_1 = NearestNeighbors(n_neighbors=k, metric="cosine").fit(d1)
    knn_2 = NearestNeighbors(n_neighbors=k, metric="cosine").fit(d2)
    knn_1 = knn_1.kneighbors(return_distance=False)
    knn_2 = knn_2.kneighbors(return_distance=False)
    
    overlap = [len(set(a).intersection(b)) for a, b in zip(knn_1, knn_2)]
    return np.mean(overlap) / k

# Compute for AstroPT JWST pairs
astropt_15m_jwst = ds['astropt_15m_jwst']
astropt_95m_jwst = ds['astropt_95m_jwst']
astropt_850m_jwst = ds['astropt_850m_jwst']

score1 = MKNN_no_clip(astropt_15m_jwst, astropt_95m_jwst, 10)
score2 = MKNN_no_clip(astropt_95m_jwst, astropt_850m_jwst, 10)

print(f"AstroPT 15M vs 95M MKNN: {score1:.4f} ({score1*100:.2f}%)")
print(f"AstroPT 95M vs 850M MKNN: {score2:.4f} ({score2*100:.2f}%)")
