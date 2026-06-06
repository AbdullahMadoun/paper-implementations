# The Platonic Universe

**Paper**: [The Platonic Universe: Do Foundation Models See the Same Sky?](https://arxiv.org/abs/2509.19453)
**Authors**: Kshitij Duraphe, Michael J. Smith, Shashwat Sourav, John F. Wu
**Year**: 2025

---

## Why This Paper?

It was mentioned on the "How Do Foundation Models See the Same Sky?" project for SORA, so I reviewed it after reviewing the original Platonic Representation Hypothesis paper.

## Key Idea

This paper tests the platonic representation hypothesis for astronomical models using the same methodology. It proves that the platonic hypothesis consistently holds across models and cross-modally to spectroscopic and imaging observation datasets.

## What I Implemented

- Recreated the MKNN comparisons for cross-modal alignment across model sizes.
- Visualized the scaling convergence trends for AstroPT, ConvNeXtv2, DINOv2, IJEPA, and ViT.

## Results & Takeaways

The MKNN scores confirmed that larger models exhibit stronger representational alignment, both intramodal and crossmodal. The unique contribution of this paper is demonstrating that the Platonic Representation Hypothesis holds true for astronomical observations. By testing across different data modalities, for astronomy datasets the results show that models naturally converge toward a shared representation of the underlying astrophysics. This supports the platonic hypothesis's conclusion that as model capacity scales up, diverse neural networks learn universal structural patterns that transcend their specific training domains.

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
| `assets/` | Saved figures, embeddings, etc. |
