from datasets import load_dataset
import numpy as np

print("Loading jwst_hsc_embeddings...")
ds = load_dataset("UniverseTBD/jwst_hsc_embeddings", split="train")
print(f"Total rows: {len(ds)}")
print("Features:", list(ds.features.keys()))

# Check how many non-None elements are in astropt_15m_jwst
col1 = ds['astropt_15m_jwst']
valid = [x for x in col1 if x is not None]
print(f"Valid elements in astropt_15m_jwst: {len(valid)}")
