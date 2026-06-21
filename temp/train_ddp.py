"""
Multi-GPU DDP training script.
Run with: torchrun --nproc_per_node=NUM_GPUS train_ddp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import urllib.request

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 256   # reduced from 2048 so we have more batches on 10k tokens
BATCH_SIZE = 8
D_MODEL    = 512
N_HEADS    = 8
EPOCHS     = 1
DATA_LIMIT = 10_000   # chars to use from Shakespeare
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare.txt"):
        urllib.request.urlretrieve(url, "shakespeare.txt")
    with open("shakespeare.txt", "r") as f:
        text = f.read()

    chars     = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi      = {ch: i for i, ch in enumerate(chars)}
    encode    = lambda s: [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    data = data[:DATA_LIMIT]
    return data, vocab_size


class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx       : idx + self.seq_len]
        y = self.data[idx + 1   : idx + self.seq_len + 1]
        return x, y


# ── Model components ───────────────────────────────────────────────────────────
class Linear(nn.Module):
    def __init__(self, d_in=512, d_out=1024, activate=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_out, dtype=torch.bfloat16))
        self.b = nn.Parameter(torch.zeros(d_out,        dtype=torch.bfloat16))
        self.activate = activate

    def forward(self, x):
        out = torch.matmul(x, self.W) + self.b
        return F.relu(out) if self.activate else out


class Embedding(nn.Module):
    def __init__(self, d_in=10000, d_out=512):
        super().__init__()
        # Use a standard embedding table — much more efficient than one_hot matmul
        self.table = nn.Embedding(d_in, d_out).to(torch.bfloat16)

    def forward(self, x):
        return self.table(x)


class MLP(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(d_model,   d_model * 2, activate=True),
            Linear(d_model*2, d_model * 2, activate=True),
            Linear(d_model*2, d_model * 4, activate=True),
            Linear(d_model*4, d_model * 2, activate=True),
            Linear(d_model*2, d_model,     activate=False),
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).to(torch.bfloat16))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LayerNorm(nn.Module):
    def __init__(self, d_model=512, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones( d_model, dtype=torch.bfloat16))
        self.beta  = nn.Parameter(torch.zeros(d_model, dtype=torch.bfloat16))
        self.eps   = eps

    def forward(self, x):
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.var( dim=-1, keepdim=True)
        return self.gamma * (x - mu) / torch.sqrt(sigma + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n=8):
        super().__init__()
        assert d_model % n == 0
        self.n      = n
        self.d_model = d_model
        self.h_dim  = d_model // n
        self.K = Linear(d_model, d_model)
        self.Q = Linear(d_model, d_model)
        self.V = Linear(d_model, d_model)
        self.O = Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        K = self.K(x).view(B, T, self.n, self.h_dim).transpose(1, 2)
        Q = self.Q(x).view(B, T, self.n, self.h_dim).transpose(1, 2)
        V = self.V(x).view(B, T, self.n, self.h_dim).transpose(1, 2)

        scores = Q.matmul(K.transpose(-2, -1)) / math.sqrt(self.h_dim)
        mask   = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1).matmul(V)

        x = attn.transpose(1, 2).contiguous().flatten(2, 3)
        return self.O(x)


class GPTTransformer(nn.Module):
    def __init__(self, d_model=512, n=8, v=65, cw=256):
        super().__init__()
        self.emb         = Embedding(d_in=v, d_out=d_model)
        self.pos_enc     = PositionalEncoding(d_model=d_model, max_len=cw)
        self.ln1         = LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(d_model=d_model, n=n)
        self.ln2         = LayerNorm(d_model)
        self.mlp         = MLP(d_model)
        self.linear      = Linear(d_model, v)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_enc(x)
        x = x + self.masked_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return self.linear(x)


# ── DDP training loop ──────────────────────────────────────────────────────────
def train(rank, world_size, data, vocab_size):
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically
    init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    dataset    = CharDataset(data, SEQ_LEN)
    sampler    = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    model     = GPTTransformer(v=vocab_size, cw=SEQ_LEN).to(rank)
    model     = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.synchronize(rank)
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)

            logits = model(batch_x)
            loss   = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[GPU {rank}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Batches: {len(dataloader)}")

    torch.cuda.synchronize(rank)
    elapsed = time.time() - t0

    if rank == 0:
        print(f"\n✅ Multi-GPU ({world_size} GPUs) | Total time: {elapsed:.2f}s")

    destroy_process_group()


if __name__ == "__main__":
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    data, vocab_size = load_data()
    train(local_rank, world_size, data, vocab_size)
