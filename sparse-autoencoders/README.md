# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

**Authors**: Anthropic (Trenton Bricken et al.)  
**Year**: 2023  
**Link**: [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)

## Why This Topic?
I just found out about SAEs from the Anthropic paper "Towards Monosemanticity". After watching a few videos to gain intuition, I tried to train one with the hypothesis that I may find features that directly map to labeled emotions. The phenomenology and universality of features caught my attention, and I wanted to explore it further.

## Key Idea
The key idea is to resolve "polysemanticity" (where neurons in neural networks represent multiple unrelated concepts due to superposition) by using Sparse Autoencoders (SAEs). An SAE reconstructs the internal activations of a language model while heavily penalizing non-sparse activations. This forces the autoencoder to learn an overcomplete dictionary of features where each feature (or direction) is sparsely activated and monosemantic (highly interpretable).

## What I Implemented
I implemented:
- A Sparse Autoencoder with an encoder (linear + ReLU) and a linear decoder.
- The standard training loop using an L1 penalty on the hidden activations and an MSE reconstruction loss.
- Post-activation collection balancing by randomly oversampling minority classes to ensure balanced training data.
- Extraction of features from the trained SAE.

**Personal Annotations**:
- **Learning PyTorch Hooks**: This was my very first time learning about PyTorch hooks and capturing intermediate activations from a live transformer. I wanted to emphasize building the base model loading and hooking logic entirely myself. Figuring out how to intercept the hidden states (specifically from `model.transformer.h[6]`) during the forward pass
- **Feature-Level Interpretability**: This was also my first time interacting with feature-level interactions! Looking at raw, high-dimensional activation arrays and attempting to decipher them is daunting but incredibly rewarding.
- **Dataset Selection & Context Windows**: I experimented with different datasets. My primary goal was to find a dataset where I could pass entire texts without aggressive clipping so they would fit entirely within the context window. I specifically wanted to use a labeled dataset to see if the SAE natively captures those human-defined labels (like emotions).
- **Feature Exploration**: For the feature exploration phase, I tried a little manual exploration myself but didn't have much luck finding easily interpretable directions immediately. To assist, I utilized an LLM to search for and annotate specific features using my activation setup, which revealed some fascinating semantic circuits.

## Results & Takeaways
The model successfully identifies specific features that activate strongly for specific text patterns (e.g., anxiety/fear). The oversampling ensured the autoencoder didn't just learn to reconstruct majority class tokens. 

**Key Insight:** Initially, extracting exact features mapping one-to-one per emotion was challenging. To remove any bias from the unbalanced dataset labels, a DataLoader with oversampling was implemented. The difficulty in extracting exact features per emotion is likely because GPT-2 is relatively lightweight and may not have the capacity to generalize that deeply into complex emotions.

You can view the interactive results and visualizations here: [Interpretability Showcase](./visuals/interpretability_showcase.html).
