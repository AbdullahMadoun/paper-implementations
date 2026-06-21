import subprocess
import sys

print("Spawning background notebook execution...")

script = """
import os
import sys

os.system('pip install -q nbclient nbformat datasets transformers safetensors tqdm')

import nbformat
from nbclient import NotebookClient

try:
    with open('/content/experiment.ipynb') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=3600, kernel_name='python3')
    client.execute()

    with open('/content/experiment.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    with open('/content/done.txt', 'w') as f:
        f.write('done')
except Exception as e:
    with open('/content/error.txt', 'w') as f:
        f.write(str(e))
    with open('/content/done.txt', 'w') as f:
        f.write('error')
"""

with open('/content/run_nb.py', 'w') as f:
    f.write(script)

subprocess.Popen(["nohup", "python3", "/content/run_nb.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)

print("Background execution spawned successfully.")
sys.exit(0)
