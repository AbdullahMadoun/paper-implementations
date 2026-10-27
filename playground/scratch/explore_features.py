import argparse
import torch
import sys
from src.data import load_base_model, get_text_dataset
from src.pretrained import load_pretrained_sae
from src.explore import FeatureExplorer

def parse_args():
    parser = argparse.ArgumentParser(description="Explore Features of a Pre-trained or Custom SAE")
    parser.add_argument("--layer_idx", type=int, default=6, help="Layer index of the SAE (0-11)")
    parser.add_argument("--repo_id", type=str, default="jbloom/GPT2-Small-SAEs-Reformatted", help="HF Repo ID for pre-trained weights")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k", help="Dataset to build database from")
    parser.add_argument("--num_docs", type=int, default=200, help="Number of documents to scan for top contexts")
    parser.add_argument("--custom_sae_path", type=str, default=None, help="Path to a custom trained SAE .pt checkpoint")
    return parser.parse_args()

def interactive_loop(explorer: FeatureExplorer, layer_idx: int):
    print("\n" + "="*80)
    print("INTERACTIVE SAE FEATURE EXPLORER")
    print("="*80)
    print("Commands:")
    print("  - Enter an integer (e.g., '47') to view top-activating contexts for that feature.")
    print("  - Enter a sentence (e.g., 'The president visited the capital.') to analyze token activations.")
    print("  - Type 'exit' or 'quit' to close.")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("\nQuery (Feature index or sentence): ").strip()
            if not query:
                continue
                
            if query.lower() in ["exit", "quit"]:
                print("Exiting interactive explorer.")
                break
                
            # Check if query is feature index
            if query.isdigit():
                feat_idx = int(query)
                if feat_idx < 0 or feat_idx >= explorer.d_sae:
                    print(f"Error: Feature index must be between 0 and {explorer.d_sae - 1}")
                    continue
                
                print(f"\nRetrieving top-activating contexts for Feature {feat_idx}...")
                contexts = explorer.get_feature_summary(feat_idx, context_window=5)
                
                if not contexts:
                    print(f"Feature {feat_idx} did not activate on any tokens in the scanned {explorer.database['raw_docs'].__len__()} documents.")
                    print("Try entering another feature index or scanning more documents.")
                    continue
                    
                print(f"\nFound {len(contexts)} activating occurrences:")
                for rank, ctx in enumerate(contexts):
                    print(f"\nRank {rank+1} (Activation: {ctx['activation']:.4f}, Doc {ctx['doc_idx']}, Token {ctx['token_idx']}):")
                    # Print context highlighting the active token
                    tokens_copy = list(ctx["context_tokens"])
                    hl_pos = ctx["highlight_position"]
                    if 0 <= hl_pos < len(tokens_copy):
                        # Add terminal color (yellow/bold) to the highlighted token
                        tokens_copy[hl_pos] = f"\033[1;33m[{tokens_copy[hl_pos]}]\033[0m"
                    print("Context: ... " + "".join(tokens_copy) + " ...")
            else:
                # Custom sentence analysis
                print(f"\nAnalyzing custom text: '{query}'")
                analysis = explorer.analyze_custom_text(query, layer_idx=layer_idx)
                
                print("\nToken activations:")
                for token_info in analysis["analysis"]:
                    token_str = repr(token_info["token"])
                    features_str = ", ".join([f"Feat {f['feature_idx']} ({f['activation']:.2f})" for f in token_info["features"][:3]])
                    if features_str:
                        print(f"  {token_str:<12} -> {features_str}")
                    else:
                        print(f"  {token_str:<12} -> (No features active)")
                        
                print("\nTop features activated in the sentence:")
                for f_info in analysis["top_features"][:5]:
                    print(f"  Feature {f_info['feature_idx']:<5} on '{f_info['activating_token']}' (max activation: {f_info['max_activation']:.4f})")
                    
        except KeyboardInterrupt:
            print("\nExiting interactive explorer.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

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
    model, tokenizer = load_base_model("gpt2", device=device)
    
    # 2. Load SAE
    if args.custom_sae_path:
        # Load custom SAE
        print(f"Loading custom SAE checkpoint from {args.custom_sae_path}...")
        checkpoint = torch.load(args.custom_sae_path, map_location=device)
        from src.model import SparseAutoencoder
        sae = SparseAutoencoder(
            d_in=checkpoint["d_in"],
            d_sae=checkpoint["d_sae"],
            l1_coef=checkpoint["l1_coef"]
        )
        sae.load_state_dict(checkpoint["model_state_dict"])
        sae.to(device)
    else:
        # Load pre-trained SAE
        try:
            sae = load_pretrained_sae(layer_idx=args.layer_idx, repo_id=args.repo_id, device=device)
        except NotImplementedError:
            print("\n" + "!"*80)
            print("COULD NOT RUN EXPLORATION!")
            print("The pre-trained loader initializes your SparseAutoencoder class from scratch.")
            print("Please implement SparseAutoencoder in 'src/model.py' first!")
            print("!"*80 + "\n")
            return
            
    # 3. Load Dataset
    dataset = get_text_dataset(args.dataset_name)
    
    # 4. Instantiate Explorer
    explorer = FeatureExplorer(sae=sae, model=model, tokenizer=tokenizer, device=device)
    
    # 5. Build Database of Activations
    explorer.build_feature_database(
        dataset=dataset,
        layer_idx=args.layer_idx,
        num_docs=args.num_docs,
        seq_len=128,
        batch_size=16,
        top_k=10
    )
    
    # 6. Run Interactive Loop
    interactive_loop(explorer, layer_idx=args.layer_idx)

if __name__ == "__main__":
    main()
