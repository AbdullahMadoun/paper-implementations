import os
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any, List
from .model import SparseAutoencoder
from .buffer import ActivationBuffer

class SAETrainer:
    def __init__(
        self,
        sae: SparseAutoencoder,
        activation_buffer: ActivationBuffer,
        lr: float = 3e-4,
        total_steps: int = 10000,
        lr_warmup_steps: int = 500,
        l1_warmup_steps: int = 1000,
        dead_feature_window: int = 1000,
        device: str = "cpu"
    ):
        """
        Trainer for Sparse Autoencoder.
        
        Args:
            sae: The SparseAutoencoder model.
            activation_buffer: The ActivationBuffer yielding base model activations.
            lr: Peak learning rate.
            total_steps: Total number of training steps.
            lr_warmup_steps: Number of steps to warm up learning rate.
            l1_warmup_steps: Number of steps to warm up the L1 coefficient.
            dead_feature_window: Number of steps to monitor to determine if a feature is dead.
            device: PyTorch device ('cpu', 'cuda', 'mps').
        """
        self.sae = sae.to(device)
        self.buffer = activation_buffer
        self.lr = lr
        self.total_steps = total_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.l1_warmup_steps = l1_warmup_steps
        self.dead_feature_window = dead_feature_window
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(self.sae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        # Dead feature tracking
        # Counts how many steps since each feature was last active
        self.steps_since_active = torch.zeros(self.sae.d_sae, device=self.device)
        
        # Store metrics history
        self.history = []

    def get_lr_scale(self, step: int) -> float:
        """Computes learning rate scale factor with linear warmup and cosine decay."""
        if step < self.lr_warmup_steps:
            return float(step) / float(max(1, self.lr_warmup_steps))
        
        # Cosine decay from peak lr to 0
        progress = float(step - self.lr_warmup_steps) / float(max(1, self.total_steps - self.lr_warmup_steps))
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_l1_scale(self, step: int) -> float:
        """Computes L1 scale factor with linear warmup to avoid early feature death."""
        if step < self.l1_warmup_steps:
            return float(step) / float(max(1, self.l1_warmup_steps))
        return 1.0

    def step_train(self, step: int) -> Dict[str, float]:
        """Runs a single step of SAE training."""
        self.optimizer.zero_grad()
        
        # Get a batch of shuffled activations
        x = next(self.buffer_iter)
        
        # TODO: Implement the SAE training step!
        # Hints:
        # 1. Run forward pass: x_hat, feature_acts = self.sae(x)
        # 2. Get L1 scale coefficient: l1_scale = self.get_l1_scale(step)
        # 3. Compute loss: use self.sae.compute_loss(x, x_hat, feature_acts)
        #    Make sure to scale the L1 coefficient by l1_scale!
        # 4. Backward pass: loss.backward()
        # 5. Clip gradients: torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
        # 6. Adjust learning rate: set self.optimizer.param_groups[0]['lr'] = self.lr * self.get_lr_scale(step)
        # 7. Optimizer step: self.optimizer.step()
        # 8. Enforce decoder unit norm constraint: self.sae.make_decoder_weights_unit_norm()
        # 9. Track dead features:
        #    - activated_this_batch = (feature_acts > 1e-5).any(dim=0)
        #    - reset steps_since_active to 0 for active ones, increment for inactive ones.
        
        raise NotImplementedError("You need to implement the step_train method in the trainer!")


    def train(self) -> List[Dict[str, float]]:
        """Trains the SAE for the specified total_steps."""
        print(f"Starting training on {self.device} for {self.total_steps} steps...")
        self.sae.train()
        
        # Reset dead feature tracker
        self.steps_since_active.zero_all_ = self.steps_since_active.zero_()
        
        # Get iterator from the buffer
        self.buffer_iter = iter(self.buffer)
        
        # Initialize decoder bias to the mean of the first batch to speed up convergence
        # This is a standard initialization trick
        first_batch = next(self.buffer_iter)
        self.sae.b_dec.data = first_batch.mean(dim=0).clone()
        
        progress_bar = tqdm(range(self.total_steps), desc="SAE Training")
        for step in progress_bar:
            step_metrics = self.step_train(step)
            self.history.append(step_metrics)
            
            # Update progress bar description
            progress_bar.set_postfix({
                "loss": f"{step_metrics['loss']:.4f}",
                "recon": f"{step_metrics['recon_loss']:.4f}",
                "l0": f"{step_metrics['l0']:.1f}",
                "exp_var": f"{step_metrics['explained_variance']:.2%}",
                "dead_%": f"{step_metrics['pct_dead_features']:.1f}%"
            })
            
            # Print periodic logs
            if step > 0 and step % 1000 == 0:
                print(
                    f"Step {step:05d} | Loss: {step_metrics['loss']:.4f} | "
                    f"Recon: {step_metrics['recon_loss']:.4f} | L0: {step_metrics['l0']:.1f} | "
                    f"Exp Var: {step_metrics['explained_variance']:.2%} | Dead: {step_metrics['pct_dead_features']:.1f}%"
                )
                
        self.buffer.close()
        return self.history

    def save(self, path: str):
        """Saves the trained model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.sae.state_dict(),
            "d_in": self.sae.d_in,
            "d_sae": self.sae.d_sae,
            "l1_coef": self.sae.l1_coef,
            "history": self.history
        }, path)
        print(f"Model saved to {path}")
