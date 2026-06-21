# Auto-Encoding Variational Bayes

**Authors:** Diederik P. Kingma, Max Welling  
**Year:** 2013  
**Link:** [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)  

## Why This Paper?
VAEs are the foundation for learning smooth, continuous, and steerable latent representations. The complex math and intuition behind the Evidence Lower Bound (ELBO) were traced from [this lecture](https://www.youtube.com/watch?v=gErs0bAM63E&list=PLehuLRPyt1HxuYpdlW4KevYJVOSDG3DEz&index=16).

## Key Idea
Instead of mapping an image to a single rigid point in space, a VAE maps an image to a *probability distribution* (a blurry region). By forcing all these distributions to overlap near the center using KL-Divergence, the model creates a smooth, continuous "blob" of concepts. The famous "reparameterization trick" ($z = \mu + \sigma \odot \epsilon$) separates the randomness from the network weights, allowing us to successfully backpropagate gradients.

## What I Implemented
- An MLP-based Encoder that outputs $\mu$ and $\log(\sigma^2)$.
- The Reparameterization Trick.
- An MLP-based Decoder.
- Custom ELBO Loss (Summing over features, averaging over the batch to prevent Posterior Collapse).
- **Noise Distribution Study:** Used UMAP to cluster the latent space and calculate the average random noise vector ($\epsilon$) responsible for generating specific digits.

## Results & Takeaways
- **Debugging Loss Reduction:** While debugging an issue with unusually low loss values, I learned that letting `F.mse_loss` default to `reduction="mean"` averages the error across both the batch and the 784 pixels. This shrinks the reconstruction loss significantly, causing the KL-Divergence to overpower it and leading to posterior collapse. The mathematically correct approach is to respect the tensor dimensions: 
  - The input image is `[64, 784]`.
  - We first sum across the feature dimension `dim=-1` to get the total reconstruction error per image `[64]`.
  - We then take the mean across the batch dimension `dim=0` to get a stable scalar loss. 
  - This effectively keeps the reconstruction loss and KL-Divergence properly balanced.
- **Latent Steering and Averaging Noise:** In the noise study, we generated images from pure random noise and clustered them by digit. By taking the average of all the noise vectors that generated a "7" and decoding that average, the model produced a very clear, generic "7". Because the VAE is a continuous mapping, the random stylistic variations (like slant or width) cancel out when averaged. This leaves a central vector that points to the core representation of that digit in the latent space, which is basically the foundation for how we steer outputs in generative models like GANs.
