import json
import sys

def insert_markdown_cell(cells, index, source):
    return cells[:index] + [{
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }] + cells[index:]

def process_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # We want to add "My Work" before the first cell
    cells = insert_markdown_cell(cells, 0, "> ✍️ **My Work** — Reimplementing embeddings and exploring gather/sparse functionality.")
    
    # The last code cell contains run_four_way_benchmark. Let's find it.
    idx = 0
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code' and any('run_four_way_benchmark' in line for line in c.get('source', [])):
            idx = i
            break
            
    if idx > 0:
        cells = insert_markdown_cell(cells, idx, "> 🤖 **AI-Assisted** — Generating the benchmarking loop for performance comparison.")
        
    nb['cells'] = cells
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    process_notebook('/Users/abdullah/Pytorch/sparse-gather/experiment.ipynb')
