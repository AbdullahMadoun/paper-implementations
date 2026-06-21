import json
import sys

def process_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # Remove existing authorship markdown cells
    new_cells = []
    for c in cells:
        if c['cell_type'] == 'markdown':
            src = ''.join(c.get('source', []))
            if 'My Work' in src or 'AI-Assisted' in src:
                continue # Skip this cell
        new_cells.append(c)
        
    cells = new_cells
    
    # Now insert markdown before each code cell
    final_cells = []
    for c in cells:
        if c['cell_type'] == 'code':
            src = ''.join(c.get('source', []))
            marker = ""
            if 'import torch' in src:
                marker = "> 🤖 **AI-Assisted** — Setup and boilerplate imports."
            elif 'class GatherEmbedding' in src:
                marker = "> ✍️ **My Work** — Figuring out how to use `torch.gather` for embeddings."
            elif 'class SparseEmbedding' in src:
                marker = "> ✍️ **My Work** — Figuring out `torch.sparse_coo` for sparse operations."
            elif 'class OneHotEmbedding' in src:
                marker = "> ✍️ **My Work** — Reimplementing one-hot dense operations for baseline comparison."
            elif 'class NativeEmbedding' in src:
                marker = "> ✍️ **My Work** — Reimplementing native indexing the right way."
            elif 'def run_four_way_benchmark' in src:
                marker = "> 🤖 **AI-Assisted** — Generating the benchmarking loop for performance comparison."
                
            if marker:
                final_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [marker]
                })
        final_cells.append(c)
        
    nb['cells'] = final_cells
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    process_notebook('/Users/abdullah/Pytorch/sparse-gather/experiment.ipynb')
