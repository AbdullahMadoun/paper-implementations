import json
import os

notebook_path = "/Users/abdullah/Pytorch/multi-gpu-transformer/experiment.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

annotations = {
    # Match by a snippet of the source code to reliably find the right cell
    "import torch \n": "### 1. Standard PyTorch Imports\nImporting the core PyTorch libraries needed for building the neural network components from scratch.",
    "#multi gpu imports": "### 2. Distributed Training Imports\nImporting PyTorch's multiprocessing and Distributed Data Parallel (DDP) modules. These are essential for scaling the model across multiple GPUs.",
    "import urllib.request": "### 3. Data Loading & Tokenization\nDownloading the Tiny Shakespeare dataset and creating a custom PyTorch `Dataset` and `DataLoader`. Notice the character-level tokenization (`vocab_size = 65`) to keep things simple.",
    "device = 0": "### 4. Device Selection\nSetting the default device to GPU 0 for the single-GPU baseline.",
    "class Linear(nn.Module):": "### 5. Custom Linear Layer\nImplementing a standard linear (dense) layer using raw `nn.Parameter` tensors for weights and biases, instead of using `nn.Linear`.",
    "class Embedding(nn.Module):": "### 6. Custom Embedding Layer\nAn embedding layer implemented by converting input indices to one-hot vectors and multiplying by a weight matrix.",
    "class MLP(nn.Module):": "### 7. Multi-Layer Perceptron (MLP)\nThe feed-forward network used inside each Transformer block, consisting of multiple linear layers with ReLU activations.",
    "class PositionalEncoding(nn.Module):": "### 8. Positional Encoding\nSince Transformers process all tokens in parallel, they lack an inherent sense of sequence order. This injects sine and cosine frequencies so the model knows where each token is positioned.",
    "class LayerNorm(nn.Module):": "### 9. Custom Layer Normalization\nImplementing LayerNorm from scratch by computing the mean and variance across the embedding dimension, then shifting and scaling with learnable parameters `gamma` and `beta`.",
    "class MultiHeadAttention(nn.Module):": "### 10. Multi-Head Causal Attention\nThe core of the Transformer! This computes scaled dot-product attention. Notice the `causal_mask` which prevents tokens from 'looking ahead' into the future during autoregressive training.",
    "class GPTTransformer(nn.Module):": "### 11. Full Transformer Assembly\nBringing it all together: Embedding + Positional Encoding -> LayerNorm -> Masked Attention -> LayerNorm -> MLP -> Output Projection.",
    "model = GPTTransformer()": "### 12. Model Instantiation\nCreating the model and moving it to the target device.",
    "optimizer = torch.optim.AdamW": "### 13. Optimizer Setup\nUsing AdamW for gradient descent optimization.",
    "EPOCHS = 1\n\nfor i in range(EPOCHS):": "### 14. Single GPU Training Loop\nA standard training loop computing cross-entropy loss, backpropagating gradients, and updating weights. This serves as our baseline benchmark.",
    "def ddp_setup(rank": "### 15. DDP Initialization\nSetting up the distributed process group using the `nccl` backend. This is required before any distributed communication can happen.",
    "def run ( rank, world_size):": "### 16. Distributed Training Loop\nThe training function that will be executed independently on each GPU. Notice the use of `DistributedSampler` to ensure each GPU gets a distinct slice of the dataset.",
    "mp.start_processes(run": "### 17. Multi-Processing Execution\nSpawning the DDP processes across all available GPUs. We synchronize before and after to get an accurate measurement of the wall-clock time."
}

new_cells = []

for cell in nb["cells"]:
    # Check if the cell is a code cell
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        matched_annotation = None
        
        # Find which annotation matches this cell
        for key, text in annotations.items():
            if key in source_text:
                matched_annotation = text
                break
        
        # If we found an annotation, insert a markdown cell right before it
        if matched_annotation:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [matched_annotation]
            })
            
    # Always append the original cell (whether code or markdown)
    new_cells.append(cell)

nb["cells"] = new_cells

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Successfully annotated notebook cells!")
