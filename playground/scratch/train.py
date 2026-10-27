import argparse
import torch
import os
from src.data import load_base_model, get_text_dataset
from src.model import SparseAutoencoder
from src.buffer import ActivationBuffer
from src.trainer import SAETrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder on LLM Activations")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base LLM to hook")
    parser.add_argument("--layer_idx", type=int, default=6, help="Layer to hook (0-11 for gpt2)")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k", help="Dataset to load")
    parser.add_argument("--expansion_factor", type=int, default=4, help="SAE dimension expansion (d_sae = expansion * d_in)")
    parser.add_argument("--l1_coef", type=float, default=3e-4, help="L1 sparsity coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--total_steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--buffer_size", type=int, default=200_000, help="Number of activation vectors in buffer")
    parser.add_argument("--sae_batch_size", type=int, default=4096, help="SAE training batch size")
    parser.add_argument("--model_batch_size", type=int, default=16, help="LLM inference batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Token sequence length")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save weights")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Base Model and Tokenizer
    model, tokenizer = load_base_model(args.model_name, device=device)
    d_in = model.config.n_embd
    d_sae = d_in * args.expansion_factor
    print(f"Activation dimension (d_in): {d_in} | SAE hidden dimension (d_sae): {d_sae}")
    
    # 2. Load Dataset
    dataset = get_text_dataset(args.dataset_name)
    
    # 3. Initialize Activation Buffer
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layer_idx=args.layer_idx,
        buffer_size=args.buffer_size,
        model_batch_size=args.model_batch_size,
        sae_batch_size=args.sae_batch_size,
        seq_len=args.seq_len,
        device=device
    )
    
    # 4. Instantiate custom SAE
    print("Initializing Sparse Autoencoder...")
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, l1_coef=args.l1_coef)
    
    # 5. Initialize Trainer
    trainer = SAETrainer(
        sae=sae,
        activation_buffer=buffer,
        lr=args.lr,
        total_steps=args.total_steps,
        lr_warmup_steps=int(args.total_steps * 0.1),
        l1_warmup_steps=int(args.total_steps * 0.1),
        dead_feature_window=1000,
        device=device
    )
    
    # 6. Train the SAE
    try:
        history = trainer.train()
    except NotImplementedError as e:
        print("\n" + "!"*80)
        print("TRAINING NOT STARTED!")
        print("You need to implement the SparseAutoencoder model in 'src/model.py'")
        print("and the 'step_train' method in 'src/trainer.py' first!")
        print("!"*80 + "\n")
        return
        
    # 7. Save model
    save_path = os.path.join(args.save_dir, f"sae_{args.model_name}_layer{args.layer_idx}.pt")
    trainer.save(save_path)
    
if __name__ == "__main__":
    main()
