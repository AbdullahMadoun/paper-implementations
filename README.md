# 📑 Paper Experiments & Implementations

My personal sandbox for exploring AI research. The goal isn't to build the full paper, but to quickly code up core mechanics, test ideas, and build intuition.

## 📋 Papers

| # | Paper | Folder | Key Idea | Status |
|---|-------|--------|----------|--------|
| 1 | [Relative Representations (Moschella et al., 2022)](https://arxiv.org/abs/2209.15430) | [`relative-representations/`](./relative-representations/) | Cross-model latent alignment via anchor similarities | ✅ Done |
| 2 | [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) | [`vision-transformer/`](./vision-transformer/) | Patch-based transformer for image classification | 🚧 WIP |
| 3 | [UMAP: Uniform Manifold Approximation and Projection (McInnes et al., 2018)](https://arxiv.org/abs/1802.03426) | [`umap/`](./umap/) | Dimension reduction via fuzzy simplicial set topology and cross-entropy optimization | ✅ Done |

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
