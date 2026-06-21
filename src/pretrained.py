import os
import json
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from .model import SparseAutoencoder

def load_pretrained_sae(
    layer_idx: int = 6,
    repo_id: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    device: str = "cpu"
) -> SparseAutoencoder:
    """
    Downloads and loads a pre-trained residual stream SAE for GPT-2 small from Hugging Face.
    Matches the weights directly to your custom SparseAutoencoder class.
    
    Args:
        layer_idx: Transformer layer (0 to 11). Layer 6 is a good middle layer.
        repo_id: HF Hub repository containing the reformatted weights.
        device: PyTorch device to load the model on.
    """
    print(f"Downloading pre-trained SAE weights for layer {layer_idx} from '{repo_id}'...")
    
    folder = f"blocks.{layer_idx}.hook_resid_pre"
    
    try:
        # Download weights and config from Hugging Face
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{folder}/sae_weights.safetensors")
        cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{folder}/cfg.json")
    except Exception as e:
        print(f"Failed to download layer {layer_idx} SAE: {e}.")
        print("Make sure the layer and repository are correct. Let's try downloading block 6 as fallback...")
        folder = "blocks.6.hook_resid_pre"
        weights_path = hf_hub_download(repo_id=repo_id, filename=f"{folder}/sae_weights.safetensors")
        cfg_path = hf_hub_download(repo_id=repo_id, filename=f"{folder}/cfg.json")
        
    # Read the configuration
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
        
    d_in = cfg["d_in"]
    d_sae = cfg["d_sae"]
    l1_coef = cfg.get("l1_coefficient", 3e-4)
    
    print(f"Initializing custom SparseAutoencoder with config: d_in={d_in}, d_sae={d_sae}, l1_coef={l1_coef}")
    
    # Initialize the user's custom SAE
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, l1_coef=l1_coef)
    
    # Load weights
    print(f"Loading weights from {weights_path}...")
    try:
        state_dict = load_file(weights_path)
    except Exception as e:
        print(f"Error reading safetensors file: {e}")
        raise e
        
    # Map weights to custom class parameters
    try:
        sae.W_enc.data = state_dict["W_enc"].to(device)
        sae.b_enc.data = state_dict["b_enc"].to(device)
        sae.W_dec.data = state_dict["W_dec"].to(device)
        sae.b_dec.data = state_dict["b_dec"].to(device)
    except AttributeError as e:
        print("\n" + "="*80)
        print("ATTRIBUTION ERROR: Could not map weights to the SparseAutoencoder model!")
        print("This happens because the required parameters have not been initialized in your class.")
        print("Please ensure your SparseAutoencoder.__init__ defines the following parameters:")
        print("  - self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))")
        print("  - self.b_enc = nn.Parameter(torch.zeros(d_sae))")
        print("  - self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))")
        print("  - self.b_dec = nn.Parameter(torch.zeros(d_in))")
        print("="*80 + "\n")
        raise e
        
    print(f"Successfully loaded pre-trained SAE for layer {layer_idx}!")
    return sae
