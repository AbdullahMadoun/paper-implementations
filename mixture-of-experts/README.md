# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

**Paper**: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
**Authors**: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
**Year**: 2017

---

## Why This Paper?

Following the sparse-gather experiments, I set out to build the actual Mixture-of-Experts layer. It was a tough implementation that required me to dive deep into advanced PyTorch tricks to handle the dynamic routing correctly.

## Key Idea

A full PyTorch implementation of the sparsely-gated Mixture-of-Experts layer. Instead of relying heavily on `scatter` and `gather` operations, it leverages masking. This dynamic routing strategy groups tokens assigned to an expert via boolean masks, which turned out to be much neater and faster for this level of implementation. Of course, according to Gemini and the last experiment I did in `sparse-gather`, simple masking was the most performant. To quote Gemini: "it is optimized on a lower level than just using scatter/gather."

## Implementation Deep-Dive

Beyond just swapping `scatter`/`gather` for boolean masks, I really had to wrestle with some PyTorch quirks to get this right. Here’s the fun stuff:

1. **Noisy Top-K Gating ($H(x)$):**
   The router doesn't just output raw logits; it actively injects learnable noise for exploration. I implemented this by calculating `noise_term = torch.randn_like(router_logits) * F.softplus(self.W_noise(x))` and adding it to the base logits. We initialize `W_noise` to zero so it starts deterministic and slowly learns how much noise to inject.
2. **The $(K+1)$ Trick:**
   To calculate the complex Load Balancing Loss later, you actually need the threshold value that determines if an expert "made the cut". I pull `topk(..., k=k+1)` so I have both the $K$-th and $(K+1)$-th values available for the CDF math later, and then slice off the last element `[..., :-1]` to build the actual routing gate $G(x)$.
3. **Shape Shifting (Dynamic Mask Routing):**
   The hardest part of dynamic routing was figuring out how to pass exactly the right tokens to the right experts. I ended up flattening the batch and sequence dimensions, mapping the routing tensor from `(B, seq_len, n)` into a transposed boolean mask `(n, B*seq_len)`. This let me iterate over the experts and just grab `tokens_for_expert = x_for_gather[mask_for_expert]` to process them in bulk without massive, slow tensor indexing loops. Super clean.
4. **Importance Loss ($I(X)$):**
   To prevent "expert collapse" (where one expert gets all the traffic), the network needs an importance loss. This is simply the Coefficient of Variation squared ($CV^2$) calculated over the sum of the gate weights across the batch (`G.sum(dim=(0,1))`).
5. **Load Balancing Loss ($P(X)$ & Normal CDFs):**
   This was by far the heaviest math. The load balancing loss requires estimating the probability that a token *would* be routed to an expert under the noise distribution. I used `torch.where(is_in_top_k, k_plus_1th, kth)` to perfectly replicate the paper's tricky exclusion threshold, normalized it by the standard deviation of the noise (`sp_noise`), and passed it through `torch.distributions.Normal(0, 1).cdf()` to get the probability $P(x, i)$.

## Results & Takeaways

Honestly, this was a tough build. It involved a bunch of advanced PyTorch tricks, but figuring out that masks are way cleaner and faster than scatter/gather made it an extremely rewarding experiment. 

The biggest "Aha!" moment was definitely getting those masks to navigate the dimensions properly. Wrapping my head around it was rough, but once that dimensional mapping finally clicked, the dynamic routing worked like magic. 

Translating the dense math from Appendix A into actual code was another massive hurdle. Using stuff like `torch.where` to handle cut-offs for the $k$-th vs $k+1$-th elements was completely new to me. It took a lot of searching and trial-and-error to find the most PyTorch-native way to do it.

**Next Steps:** Slapping this into a multi-GPU setup and adding formal expert capacity limits.

*(Side note: fixing that load-balancing CDF normalization bug just proved the math actually works!)*


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
