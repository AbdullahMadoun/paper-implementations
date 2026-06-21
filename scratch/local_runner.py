import os
import sys

print("Installing dependencies if missing...")
os.system("pip install -q nbclient nbformat datasets transformers safetensors tqdm")

import nbformat
from nbclient import NotebookClient

try:
    print("Loading notebook...")
    with open("sparse-autoencoders/experiment.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    print("Executing notebook locally on CPU...")
    client = NotebookClient(nb, timeout=3600, kernel_name='python3')
    client.execute()

    print("Saving executed notebook...")
    with open("sparse-autoencoders/experiment.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook execution and saving successful!")
except Exception as e:
    print(f"Error executing notebook: {e}")
    sys.exit(1)
