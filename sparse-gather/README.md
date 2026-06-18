# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

**Paper**: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
**Authors**: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
**Year**: 2017

---

## Why This Paper?

I was exploring Mixture-of-Experts (MoE) and realized that to really benefit from the architecture, we must take full advantage of sparsity by not calculating zero-values. That took me into a deep rabbit hole discovering sparse vectors in PyTorch, `scatter`, and `gather`. I also wanted to revisit my old implementation of embeddings in the `multi-gpu-transformer` project to reimplement it the right way.

## Key Idea

I went through the pain of writing the code to learn these concepts—from one-hot dense operations to sparse math and manual gather. The goal is to fully understand how PyTorch handles sparse operations before moving on to the next steps, which involve building a full MoE layer.

## What I Implemented

- **Four embedding paradigms**: `OneHotEmbedding` (dense), `SparseEmbedding` (using `torch.sparse_coo`), `GatherEmbedding` (using `torch.gather`), and `NativeEmbedding` (direct indexing).
- Re-implemented embeddings to compare the efficiency of direct indexing versus sparse matrices and gather operations.

## Results & Takeaways

I benchmarked the forward and backward passes of all four approaches. The results summarize why avoiding zero-calculations is crucial:
- **Native Indexing** is the fastest (0.0004s forward / 0.0021s backward).
- **Sparse Math** and **Manual Gather** are close (around 0.0010s forward), proving that leveraging sparsity or gathering avoids huge computational waste.
- **One-Hot Dense** is horribly slow (0.0812s forward / 0.1268s backward) because it computes massive matrices of zeros.

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
