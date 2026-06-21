import json
import os

notebook_path = "/Users/abdullah/Pytorch/multi-gpu-transformer/experiment.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

overview_markdown = """# Distributed PyTorch & Transformer Implementation

This notebook contains a from-scratch implementation of a Transformer language model (`GPTTransformer`), focusing on the core mechanics without the complexity of production models (e.g. no KV caching, dropout, or mixed precision). 

Additionally, it benchmarks the training process on a single GPU versus a multi-GPU setup using PyTorch's Distributed Data Parallel (`DDP`).

### Authorship Markers
To maintain transparency on what was hand-written versus what was assisted by AI, the sections are annotated with the following markers:
- `> ✍️ **My Work**` — The core architecture (Linear, Embedding, MLP, LayerNorm, MultiHeadAttention, GPTTransformer) which I implemented myself to build intuition.
- `> 🤖 **AI-Assisted**` — Boilerplate code such as imports, data loading, positional encoding, and the distributed computing logic (adapted from a YouTube tutorial)."""

overview_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [overview_markdown]
}

# Insert at the very top
nb["cells"].insert(0, overview_cell)

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Successfully added overview markdown to the top of the notebook!")
