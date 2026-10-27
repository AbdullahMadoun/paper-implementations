import json

notebook_path = "/Users/abdullah/Pytorch/multi-gpu-transformer/experiment.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

# Keep the first overview cell
overview_cell = nb["cells"][0]
rest_of_cells = nb["cells"][1:]

# 1. Clean up previously injected markdown cells
cleaned_cells = []
for cell in rest_of_cells:
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"]).strip()
        # Remove my previous injections
        if text.startswith("### ") or text.startswith("> 🤖") or text.startswith("> ✍️") or text.startswith("> \U0001f916"):
            continue
    cleaned_cells.append(cell)

# 2. Define the new annotations with authorship markers
annotations = {
    "import torch \n": "> 🤖 **AI-Assisted**\n\n### 1. Standard PyTorch Imports\nImporting the core PyTorch libraries needed for building the neural network components from scratch.",
    "#multi gpu imports": "> 🤖 **AI-Assisted**\n\n### 2. Distributed Training Imports\nImporting PyTorch's multiprocessing and Distributed Data Parallel (DDP) modules. These are essential for scaling the model across multiple GPUs.",
    "import urllib.request": "> 🤖 **AI-Assisted**\n\n### 3. Data Loading & Tokenization\nDownloading the Tiny Shakespeare dataset and creating a custom PyTorch `Dataset` and `DataLoader`. Notice the character-level tokenization (`vocab_size = 65`) to keep things simple.",
    "device = 0": "> ✍️ **My Work**\n\n### 4. Device Selection\nSetting the default device to GPU 0 for the single-GPU baseline.",
    "class Linear(nn.Module):": "> ✍️ **My Work**\n\n### 5. Custom Linear Layer\nImplementing a standard linear (dense) layer using raw `nn.Parameter` tensors for weights and biases, instead of using `nn.Linear`.",
    "class Embedding(nn.Module):": "> ✍️ **My Work**\n\n### 6. Custom Embedding Layer\nAn embedding layer implemented by converting input indices to one-hot vectors and multiplying by a weight matrix.",
    "class MLP(nn.Module):": "> ✍️ **My Work**\n\n### 7. Multi-Layer Perceptron (MLP)\nThe feed-forward network used inside each Transformer block, consisting of multiple linear layers with ReLU activations.",
    "class PositionalEncoding(nn.Module):": "> 🤖 **AI-Assisted**\n\n### 8. Positional Encoding\nSince Transformers process all tokens in parallel, they lack an inherent sense of sequence order. This injects sine and cosine frequencies so the model knows where each token is positioned.",
    "class LayerNorm(nn.Module):": "> ✍️ **My Work**\n\n### 9. Custom Layer Normalization\nImplementing LayerNorm from scratch by computing the mean and variance across the embedding dimension, then shifting and scaling with learnable parameters `gamma` and `beta`.",
    "class MultiHeadAttention(nn.Module):": "> ✍️ **My Work**\n\n### 10. Multi-Head Causal Attention\nThe core of the Transformer! This computes scaled dot-product attention. Notice the `causal_mask` which prevents tokens from 'looking ahead' into the future during autoregressive training.",
    "class GPTTransformer(nn.Module):": "> ✍️ **My Work**\n\n### 11. Full Transformer Assembly\nBringing it all together: Embedding + Positional Encoding -> LayerNorm -> Masked Attention -> LayerNorm -> MLP -> Output Projection.",
    "model = GPTTransformer()": "> ✍️ **My Work**\n\n### 12. Model Instantiation\nCreating the model and moving it to the target device.",
    "optimizer = torch.optim.AdamW": "> ✍️ **My Work**\n\n### 13. Optimizer Setup\nUsing AdamW for gradient descent optimization.",
    "EPOCHS = 1\n\nfor i in range(EPOCHS):": "> ✍️ **My Work**\n\n### 14. Single GPU Training Loop\nA standard training loop computing cross-entropy loss, backpropagating gradients, and updating weights. This serves as our baseline benchmark.",
    "def ddp_setup(rank": "> 🤖 **AI-Assisted**\n\n### 15. DDP Initialization\nSetting up the distributed process group using the `nccl` backend. This is required before any distributed communication can happen.",
    "def run ( rank, world_size):": "> 🤖 **AI-Assisted**\n\n### 16. Distributed Training Loop\nThe training function that will be executed independently on each GPU. Notice the use of `DistributedSampler` to ensure each GPU gets a distinct slice of the dataset.",
    "mp.start_processes(run": "> 🤖 **AI-Assisted**\n\n### 17. Multi-Processing Execution\nSpawning the DDP processes across all available GPUs. We synchronize before and after to get an accurate measurement of the wall-clock time."
}

# 3. Re-inject clean markers before every code cell
final_cells = [overview_cell]

for cell in cleaned_cells:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        matched_annotation = None
        
        for key, text in annotations.items():
            if key in source_text:
                matched_annotation = text
                break
        
        if matched_annotation:
            final_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [matched_annotation]
            })
            
    final_cells.append(cell)

nb["cells"] = final_cells

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Successfully applied authorship markers and explanations to every code cell!")
