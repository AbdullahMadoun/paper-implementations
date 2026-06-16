# 📑 Paper Experiments & Implementations

My personal sandbox for exploring AI research. The goal isn't to build the full paper, but to quickly code up core mechanics, test ideas, and build intuition.

## 📋 Papers

| # | Paper | Folder | Key Idea | Status |
|---|-------|--------|----------|--------|
| 1 | [Relative Representations (Moschella et al., 2022)](https://arxiv.org/abs/2209.15430) | [`relative-representations/`](./relative-representations/) | Cross-model latent alignment via anchor similarities | ✅ Done |
| 2 | [UMAP: Uniform Manifold Approximation and Projection (McInnes et al., 2018)](https://arxiv.org/abs/1802.03426) | [`umap/`](./umap/) | Dimension reduction via fuzzy simplicial set topology and cross-entropy optimization | ✅ Done |
| 3 | [The Platonic Universe (Duraphe et al., 2025)](https://arxiv.org/abs/2509.19453) | [`the-platonic-universe/`](./the-platonic-universe/) | Convergence of foundation models across astronomical modalities | ✅ Done |
| 4 | [Towards Monosemanticity (Bricken et al., 2023)](https://transformer-circuits.pub/2023/monosemantic-features) | [`sparse-autoencoders/`](./sparse-autoencoders/) | SAE implementation to resolve polysemanticity and find monosemantic features | ✅ Done |
| 5 | [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | [`multi-gpu-transformer/`](./multi-gpu-transformer/) | From-scratch Transformer with multi-GPU DistributedDataParallel (DDP) benchmarking & roofline analysis | ✅ Done |

## 📂 Structure

Each paper gets a standalone folder with its own notebook, README, and dependencies:

```
paper-implementations/
├── relative-representations/     
│   ├── README.md                 # Context & takeaways
│   ├── experiment.ipynb          # Implementation & results
│   └── requirements.txt          
├── vision-transformer/           
├── playground/                   # Scratchpad for learning
└── TEMPLATE/                     # Boilerplate for new papers
```

## 🤖 Authorship

I use AI heavily for boilerplate, data pipelines, and setup. To stay transparent, notebooks are tagged per-section:
- `> ✍️ **My Work**` — The core logic, experimental design, and analysis I wrote myself.
- `> 🤖 **AI-Assisted**` — Scaffolding generated or co-written with AI.

## 🚀 Run It

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r <paper-folder>/requirements.txt
```
