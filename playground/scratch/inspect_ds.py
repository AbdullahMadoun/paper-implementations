from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStoriesInstruct", split="train[:10]")
print("ROW 0:")
print(repr(ds['text'][0]))
print("\nROW 1:")
print(repr(ds['text'][1]))
