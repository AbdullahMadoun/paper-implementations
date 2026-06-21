import json

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
    
    final_cells = []
    for c in cells:
        if c['cell_type'] == 'code':
            marker = "> ✍️ **My Work**"
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
    process_notebook('/Users/abdullah/Pytorch/mixture-of-experts/experiment.ipynb')
