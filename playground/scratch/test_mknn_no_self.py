from datasets import load_dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Loading dataset...")
ds = load_dataset("UniverseTBD/jwst_hsc_embeddings", split="train")

def MKNN_exclude_self(d1, d2, k=10):
    # k+1 to exclude self
    knn_1 = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(d1)
    knn_2 = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(d2)
    dist1, ind1 = knn_1.kneighbors()
    dist2, ind2 = knn_2.kneighbors()

    matches = []
    for i in range(ind1.shape[0]):
        # remove self
        n1 = ind1[i][1:] if ind1[i][0] == i else [x for x in ind1[i] if x != i][:k]
        n2 = ind2[i][1:] if ind2[i][0] == i else [x for x in ind2[i] if x != i][:k]
        matches.append(len(np.intersect1d(n1, n2)))
    return np.sum(matches) / (float(ind1.shape[0]) * k)

print("AstroPTv2 Small JWST vs HSC (expected ~5.44%):")
score = MKNN_exclude_self(ds["astropt_15m_jwst"], ds["astropt_15m_hsc"], 10)
print(f"Score: {score*100:.2f}%")

