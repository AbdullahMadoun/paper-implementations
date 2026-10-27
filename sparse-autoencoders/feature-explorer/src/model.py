import math
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, l1_coef: float = 3e-4):
        """
        Sparse Autoencoder (SAE) skeleton.
        
        Args:
            d_in: Dimension of the base model's activations (e.g., 768 for GPT-2 small).
            d_sae: Dimension of the sparse hidden layer (expansion factor * d_in).
            l1_coef: Sparsity penalty weighting coefficient.
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coef = l1_coef
        
        # TODO: Initialize weights and biases
        # Hint:
        # - W_enc (encoder weights): shape (d_in, d_sae)
        # - b_enc (encoder bias): shape (d_sae,)
        # - W_dec (decoder weights): shape (d_sae, d_in)
        # - b_dec (decoder bias): shape (d_in,)
        # Use nn.Parameter for these.
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize weights using Kaiming Uniform initialization and ensure
        the decoder weights are normalized to unit L2 norm.
        """
        # TODO: Initialize parameters
        # Hint: use nn.init.kaiming_uniform_ on self.W_enc and self.W_dec.
        # Initialize biases to zero.
        # Call self.make_decoder_weights_unit_norm()
        pass
        
    @torch.no_grad()
    def make_decoder_weights_unit_norm(self):
        """
        Normalizes each dictionary vector (row in W_dec) to have unit L2 norm.
        This prevents the model from minimizing the L1 penalty by scaling down
        the activations and scaling up the decoder weights.
        """
        # TODO: Implement unit L2 norm constraint on decoder weights
        # W_dec shape is (d_sae, d_in)
        # HINT: To avoid PyTorch autograd leaf variable errors (which block in-place
        # operations on leaf variables that require gradients), operate on
        # `self.W_dec.data` (e.g. self.W_dec.data.div_(...) or self.W_dec.data.copy_(...)).
        pass

        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps base model activations to the sparse hidden activations.
        f(x) = ReLU((x - b_dec) @ W_enc + b_enc)
        """
        # TODO: Implement encoder projection
        # x: (batch_size, d_in)
        # return: (batch_size, d_sae)
        raise NotImplementedError("You need to implement the encode method!")
        
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original activations from the sparse hidden activations.
        x_hat = feature_acts @ W_dec + b_dec
        """
        # TODO: Implement decoder projection
        # feature_acts: (batch_size, d_sae)
        # return: (batch_size, d_in)
        raise NotImplementedError("You need to implement the decode method!")
        
    def forward(self, x: torch.Tensor):
        """
        Passes activations through the autoencoder.
        """
        # TODO: Implement forward pass (encode, then decode)
        # return: x_hat, feature_acts
        raise NotImplementedError("You need to implement the forward method!")
        
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, feature_acts: torch.Tensor):
        """
        Computes the reconstruction loss (MSE) and the L1 sparsity loss.
        """
        # TODO: Implement the loss calculation
        # - recon_loss = Mean Squared Error reconstruction loss
        # - sparsity_loss = L1 norm of activations (sum over features, averaged over batch)
        # - loss = recon_loss + l1_coef * sparsity_loss
        # return: loss, recon_loss, sparsity_loss
        raise NotImplementedError("You need to implement the compute_loss method!")
        
    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor, x_hat: torch.Tensor, feature_acts: torch.Tensor):
        """
        Computes interpretability and performance metrics.
        - L0: Average number of active features per sample (activation > 0).
        - Explained Variance: 1 - Var(error) / Var(data)
        """
        # L0: count of active features (activation > 0)
        l0 = (feature_acts > 1e-5).float().sum(dim=-1).mean().item()
        
        # Explained Variance
        total_var = x.var(dim=0).sum()
        error_var = (x - x_hat).var(dim=0).sum()
        if total_var > 1e-8:
            explained_var = (1.0 - error_var / total_var).item()
        else:
            explained_var = 0.0
            
        return {
            "l0": l0,
            "explained_variance": explained_var
        }

