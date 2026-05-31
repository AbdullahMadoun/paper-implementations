# Latent Space Alignment via Relative Representations

**Paper**: [Relative representations enable zero-shot latent space communication](https://arxiv.org/abs/2209.15430)
**Authors**: Moschella et al.
**Year**: 2022

---

## Why This Paper?

I came across this while reading [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987). The Platonic paper discusses convergence of representations across models — Relative Representations provides a concrete, testable mechanism for measuring that convergence. I wanted to implement the core mechanics and see if the geometric invariance claim holds with real models.

## Key Idea

Different models (e.g., Qwen-0.5B and TinyLlama-1.1B) live in entirely different latent spaces due to differences in architecture, dimensionality, and training. But the *geometric relationships* between shared concepts (how close "market" is to "money" or "business") are highly invariant across models. By measuring cosine similarities against a shared set of parallel anchor words, you can project absolute embeddings into a coordinate-free relative space — enabling zero-shot cross-model and cross-lingual alignment without any linear projections or Procrustes rotations.

## What I Implemented

- **Relative representation construction** — projecting absolute embeddings into anchor-relative coordinate spaces using cosine similarity
- **Two alignment metrics**:
  - KL Divergence over softmax-normalized probability profiles (my own extension, not from the paper)
  - Direct cosine similarity of relative representation vectors
- **Cross-lingual evaluation** — English (TinyLlama) ↔ Chinese (Qwen) word translation ranking

**Skipped**: Full stitching experiments, training-time relative representation losses, and large-scale evaluation. My goal was to verify the geometric invariance claim, not build a production pipeline.

## Results & Takeaways

- **KL Divergence** achieved an average rank of **2.10** (out of 9) — strong alignment
- **Cosine Similarity** achieved an average rank of **3.50** — decent but weaker
- Several words (market, future, nature, signal, theory) achieved **rank 0** (perfect match) with KL
- The KL approach with temperature-scaled softmax outperformed raw cosine — temperature acts as an amplifier that preserves structural differences

**Key insight**: The relative representation trick genuinely works even across languages and architectures with very different dimensionalities (896 vs 2048). The choice of metric matters — probabilistic comparison (KL) captures more structure than raw vector comparison.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook experiment.ipynb
```

## Files

| File | Description |
|------|-------------|
| `experiment.ipynb` | Main implementation notebook |
| `requirements.txt` | Python dependencies |
