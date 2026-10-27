import sys
import os
import torch
import torch.nn as nn
import math

# Make sure we can import from src and root
sys.path.append(os.path.abspath("."))

from src.model import SparseAutoencoder
from src.trainer import SAETrainer
from src.data import load_base_model, get_text_dataset
from src.buffer import ActivationBuffer
from src.pretrained import load_pretrained_sae

# Monkeypatch the SparseAutoencoder with completed methods for testing
def completed_init(self, d_in: int, d_sae: int, l1_coef: float = 3e-4):
    nn.Module.__init__(self)
    self.d_in = d_in
    self.d_sae = d_sae
    self.l1_coef = l1_coef
    self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
    self.b_dec = nn.Parameter(torch.zeros(d_in))
    self.reset_parameters()

def completed_reset_parameters(self):
    nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
    nn.init.zeros_(self.b_enc)
    nn.init.zeros_(self.b_dec)
    self.make_decoder_weights_unit_norm()

def completed_make_decoder_weights_unit_norm(self):
    norms = self.W_dec.norm(p=2, dim=1, keepdim=True)
    self.W_dec.data.div_(norms.clamp(min=1e-8))


def completed_encode(self, x: torch.Tensor) -> torch.Tensor:
    x_centered = x - self.b_dec
    return torch.relu(x_centered @ self.W_enc + self.b_enc)

def completed_decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
    return feature_acts @ self.W_dec + self.b_dec

def completed_forward(self, x: torch.Tensor):
    feature_acts = self.encode(x)
    return self.decode(feature_acts), feature_acts

def completed_compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, feature_acts: torch.Tensor):
    recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
    sparsity_loss = feature_acts.sum(dim=-1).mean()
    loss = recon_loss + self.l1_coef * sparsity_loss
    return loss, recon_loss, sparsity_loss

# Apply patches to model
SparseAutoencoder.__init__ = completed_init
SparseAutoencoder.reset_parameters = completed_reset_parameters
SparseAutoencoder.make_decoder_weights_unit_norm = completed_make_decoder_weights_unit_norm
SparseAutoencoder.encode = completed_encode
SparseAutoencoder.decode = completed_decode
SparseAutoencoder.forward = completed_forward
SparseAutoencoder.compute_loss = completed_compute_loss

# Monkeypatch trainer step_train
def completed_step_train(self, step: int):
    self.optimizer.zero_grad()
    x = next(self.buffer_iter)
    x_hat, feature_acts = self.sae(x)
    l1_scale = self.get_l1_scale(step)
    loss, recon_loss, sparsity_loss = self.sae.compute_loss(x, x_hat, feature_acts)
    # Re-scale L1 coefficient
    loss = recon_loss + self.sae.l1_coef * l1_scale * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
    lr_scale = self.get_lr_scale(step)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.lr * lr_scale
    self.optimizer.step()
    self.sae.make_decoder_weights_unit_norm()
    activated_this_batch = (feature_acts > 1e-5).any(dim=0)
    self.steps_since_active[activated_this_batch] = 0
    self.steps_since_active[~activated_this_batch] += 1
    num_dead = (self.steps_since_active > self.dead_feature_window).sum().item()
    pct_dead = (num_dead / self.sae.d_sae) * 100.0
    metrics = self.sae.compute_metrics(x, x_hat, feature_acts)
    return {
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "l0": metrics["l0"],
        "explained_variance": metrics["explained_variance"],
        "pct_dead_features": pct_dead,
        "lr": self.optimizer.param_groups[0]['lr'],
        "l1_coef": self.sae.l1_coef * l1_scale
    }

SAETrainer.step_train = completed_step_train

print("1. RUNNING MODEL UNIT TESTS ON COMPLETED IMPLEMENTATION IN-MEMORY...")
from test_sae_implementation import test_sae
test_sae()

print("\n2. TESTING LOADING PRE-TRAINED SAE (DOWNLOADS & CONFIG MAPPING)...")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
sae = load_pretrained_sae(layer_idx=6, device=device)
print(f"Pre-trained SAE loaded successfully! Parameters: W_enc={sae.W_enc.shape}, W_dec={sae.W_dec.shape}")

print("\n3. TESTING DUMMY DATA LOADING & ACTIVATION EXTRACTION FLOW (3 STEPS)...")
model, tokenizer = load_base_model("gpt2", device=device)
dataset = get_text_dataset("NeelNanda/pile-10k")

# Shrink buffer and seq_len for rapid validation run
buffer = ActivationBuffer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    layer_idx=6,
    buffer_size=1000,
    model_batch_size=2,
    sae_batch_size=32,
    seq_len=64,
    device=device
)

sae_small = SparseAutoencoder(d_in=768, d_sae=1536)
trainer = SAETrainer(
    sae=sae_small,
    activation_buffer=buffer,
    lr=3e-4,
    total_steps=3,
    lr_warmup_steps=1,
    l1_warmup_steps=1,
    dead_feature_window=100,
    device=device
)

history = trainer.train()
print("\nTraining flow works perfectly! Metrics history:")
for idx, step_metrics in enumerate(history):
    print(f"  Step {idx+1}: loss={step_metrics['loss']:.4f}, recon={step_metrics['recon_loss']:.4f}, l0={step_metrics['l0']:.1f}")

print("\nVERIFICATION COMPLETE: ALL IMPORTS, PIPELINES, AND MODEL MAPPINGS WORK PERFECTLY!")
