# Distributed PyTorch & Transformer Implementations

**Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) & [YouTube Tutorial](https://www.youtube.com/watch?v=-LAtx9Q6DA8&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=3)
**Authors**: Vaswani et al.
**Year**: 2017

---

## Why This Paper?

I wanted to try distributed PyTorch and in general I am pursuing my interest in distributed computing. The YouTube video was a great resource as it demonstrated training on a rented 4x 3060 setup.

## Key Idea

The key idea here is implementing Transformers from scratch (mostly), and testing the training process both on a single GPU and distributed across multiple GPUs (DDP). This also touches upon understanding theoretical bounds like compute-bound vs communication-bound, rooflines, and arithmetic intensity when communicating between chips.

## What I Implemented

- Implemented most of the core Transformer components (`Linear`, `Embedding`, `MLP`, `LayerNorm`, `MultiHeadAttention`, `GPTTransformer`) from scratch.
- Skipped data loading and positional encoding (AI generated).
- Used the YouTube tutorial for the distributed compute/DDP logic.

## Hardware & Reproducibility

- Tested on: 4× NVIDIA RTX 3060 GPUs
- Orchestration: `torchrun`
- Environment: Local Python venv using `torch.distributed` (DDP) with the NCCL backend.

## Architectural Simplifications

To focus on core mechanics and distributed systems, this `GPTTransformer` intentionally omits several features found in production models:
- **No KV Caching**: Autoregressive decoding is not optimized.
- **No Dropout or Regularization**: Built purely to observe the forward/backward pass throughput.
- **Simple Tokenization**: Uses a character-level tokenizer (`vocab_size = 65`) instead of BPE/Tiktoken.
- **No Mixed Precision**: FLOPs and memory are strictly `bfloat16` throughout, avoiding the complexities of gradient scaling.

## Theoretical Analysis: Rooflines & DDP Overhead

The DDP benchmark resulted in a **2.43× speedup** rather than a perfect 4× linear scaling. Why?

**1. NCCL AllReduce Overhead ($T_{comms}$)**
In DDP, after each backward pass, the 4 GPUs must synchronize and average their gradients. This inter-chip communication time is bound by network bandwidth. On small models, $T_{comms}$ (communication time) is proportionally large compared to $T_{math}$ (computation time).

**2. Arithmetic Intensity & Compute Bound**
A deep learning model's arithmetic intensity is the ratio of FLOPs it performs to the bytes it communicates. For a `bfloat16` matmul to be compute-bound ($T_{math} > T_{comms}$) on most accelerators, the per-replica batch size of tokens needs to exceed a critical threshold (e.g., ~240-300 tokens). Our model is small, and the local batch size computation is dwarfed by the massive parameter synchronization overhead at the end of the backward pass.

**3. Small Dataset & Batch Size**
The dataset is only ~10k characters. With 4 GPUs, each processes just 305 batches of `batch_size=8`.
- The compute-to-communication ratio is very low.
- Start-up overhead constitutes a large fraction of the total 7.48s runtime.

To move towards a 4× speedup, we would need to push the model into the compute-bound regime by: stacking more transformer layers (more compute per sync), using a larger global dataset, and increasing the per-GPU batch size.

## Results & Takeaways

Results from training on 4× RTX 3060s vs Single GPU baseline:

| | Single GPU | 4× GPU (DDP) |
|---|---|---|
| **Wall time** | 18.21s | **7.48s** |
| **Batches processed** | 1,218 | 305 × 4 = 1,220 ✅ |
| **Speedup** | 1× | **2.43×** |

The 2.43× speedup is expected because of DDP overhead (NCCL AllReduce), a tiny dataset, and small batch sizes which lower the compute-to-communication ratio.

## How to Run

> [!WARNING]
> Jupyter Notebook lacks robust support for `mp.spawn` with DDP in this environment, which caused the distributed training cells to fail. To work around this, an LLM was used to extract the notebook code into standalone Python scripts (`train_single.py` and `train_ddp.py`) for a simple execution via `torchrun`.

```bash
pip install -r requirements.txt

# To run the single GPU baseline:
python train_single.py

# To run the distributed training across 4 GPUs:
torchrun --nproc_per_node=4 train_ddp.py
```

## Files

| File | Description |
|------|-------------|
| `experiment.ipynb` | Main implementation notebook |
| `requirements.txt` | Python dependencies |
| `assets/` | Saved figures, embeddings, etc. |
| `train_single.py` | Training script for a single GPU |
| `train_ddp.py` | Distributed training script using DDP |
