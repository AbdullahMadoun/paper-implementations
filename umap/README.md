# UMAP: Uniform Manifold Approximation and Projection

**Paper**: [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)
**Authors**: Leland McInnes, John Healy, James Melville
**Year**: 2018

---

## Why This Paper?

While working on the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) paper, I saw UMAP mentioned multiple times. I realized I wasn't totally aware of how it worked under the hood or what its exact mathematical benefits were over things like t-SNE. Instead of just using the library blindly, I decided to implement it from scratch to deeply understand the core algorithm.

## Key Idea

UMAP assumes that data is uniformly distributed on a Riemannian manifold. To approximate this manifold, it constructs a "fuzzy simplicial set" (a topological graph where edge weights represent the probability of two points being connected). It then projects this high-dimensional graph into a low-dimensional space by optimizing the layout so that the low-dimensional fuzzy simplicial set closely matches the high-dimensional one, using cross-entropy and stochastic gradient descent (attractive and repulsive forces).

## What I Implemented

> **Note on Authorship**: For this implementation, I had an AI (Gemini) explain the mathematical steps to me as abstract formulas. Armed with those formulas, I went ahead and implemented the logic fully from scratch myself. 

- **Local Connectivity ($\\rho$) & Bandwidth ($\\sigma$)**: Calculating distance to the nearest neighbor and using binary search to find the correct $\\sigma$ for the fuzzy membership function.
- **Fuzzy Simplicial Set Construction**: Calculating fuzzy edge weights and combining them using the probabilistic fuzzy union ($A + B - A \\cdot B$).
- **Low-dimensional Curve Fitting**: Using `curve_fit` to find the $a$ and $b$ hyperparameters for the low-dimensional Student t-distribution approximation $1 / (1 + a \\cdot d^{2b})$.
- **Optimization**: The stochastic gradient descent training loop using attractive and repulsive forces.

**Skipped**: Advanced optimization tricks (like negative sampling instead of full repulsive force calculation) and large-scale data structures (like Nearest Neighbor Descent). The goal was to understand the math, not build a production-ready scaler.

## Results & Takeaways: Building Intuition

To truly understand why UMAP behaves the way it does, it helps to look at the "tug-of-war" between local and global forces through two conceptual lenses:

### 1. The Manifold View: Connectivity as Density
Imagine your high-dimensional data as clouds of smoke. 
- **Local Preservation (The t-SNE side)**: If you look through a microscope at a small neighborhood, you see tight clusters and strands. UMAP captures this by forcing points within a small $k$-neighbor bubble to stay grouped—like wrapping a rubber band around local smoke particles.
- **Global Preservation (The PCA side)**: If you zoom out with a wide-angle lens, you see the broad arrangement of these clouds. UMAP preserves this global skeleton because the local anchor distances ($\\rho_i$) explicitly link these bubbles together. The algorithm learns both "these points belong together" *and* "this entire cluster sits over there."

### 2. The Spring-Graph View: The Tug-of-War
Imagine the graph edges as a physical system of springs.
- **Attractive Forces (Local Constraint)**: Springs between neighbors pull tight, pulling points together to maintain clumps.
- **Repulsive Forces (Global Constraint)**: Negative sampling enforces "personal space." It pushes random points away, effectively declaring: "I don't know our exact relationship, but we aren't in the same cloud, so stay away."

**The PCA Collapse**: If you set $k$ too high, you effectively attach heavy springs between almost every point. The graph becomes so incredibly stiff that the manifold can no longer bend to fit local structures. It flattens out into a rigid linear projection, exactly like PCA.

### Resolution vs. Fidelity
UMAP's beauty is that the objective function doesn't separate "local" and "global" code—it's the exact same math, just balancing parameters:
- **High $k$ / High `min_dist`**: The wide-angle lens. You capture the broad, global trends (PCA), but the fine local details wash out.
- **Low $k$ / Low `min_dist`**: The microscope. You capture incredibly complex local clusters (t-SNE), but lose the big-picture context.

The "Holy Grail" of UMAP is simply tuning $k$ and `min_dist` so you aren't too zoomed-in to lose context, nor too zoomed-out to lose the clusters.

## How to Run

```bash
pip install -r requirements.txt
# Then open the notebook
jupyter notebook experiment.ipynb
```

## Files

| File | Description |
|------|-------------|
| `experiment.ipynb` | Main implementation notebook |
| `requirements.txt` | Python dependencies |
