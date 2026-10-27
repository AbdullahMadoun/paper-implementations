from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("NeelNanda/pile-small-tokenized-2b", split="train[:1]")
tokens = ds['tokens'][0][:20]  # Just grab first 20 tokens

# Test GPT-2
tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
text_gpt2 = tok_gpt2.decode(tokens)

# Test GPT-NeoX
tok_neox = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
text_neox = tok_neox.decode(tokens)

print("--- GPT-2 Decode ---")
print(repr(text_gpt2))
print("\n--- GPT-NeoX Decode ---")
print(repr(text_neox))
