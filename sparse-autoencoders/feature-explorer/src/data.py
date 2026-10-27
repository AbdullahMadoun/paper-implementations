import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_base_model(model_name="gpt2", device="cpu"):
    """
    Loads the base transformer model and tokenizer.
    Locks weights to avoid accidental training and sets eval mode.
    """
    print(f"Loading base language model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer

def get_text_dataset(dataset_name="NeelNanda/pile-10k", split="train"):
    """
    Loads a text dataset from Hugging Face for tokenization.
    Default is NeelNanda/pile-10k which is standard and clean for SAE exploration.
    Fallback is 'wikitext' (wikitext-2-raw-v1) if the Pile is unavailable.
    """
    print(f"Loading dataset: {dataset_name} ({split})...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Falling back to wikitext-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Filter out empty strings
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    return dataset
