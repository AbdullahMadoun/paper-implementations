import json

with open('the-platonic-universe/experiment.ipynb') as f:
    nb = json.load(f)

markers = {
    0: "> 🤖 **AI-Assisted**",
    1: "> 🤖 **AI-Assisted**",
    2: "> ✍️ **My Work**",
    3: "> ✍️ **My Work** — This was a tough one I tried implementing KNN from scratch but got different results I then searched how to build each part indivually the matching and checking etc and cross checked my formulas on paper etc",
    4: "> ✍️ **My Work**",
    5: "> ✍️ **My Work**",
    6: "> ✍️ **My Work**",
    7: "> ✍️ **My Work**",
    8: "> ✍️ **My Work**",
    10: "> ✍️ **My Work**",
    11: "> ✍️ **My Work**",
    12: "> 🤖 **AI-Assisted**",
    13: "> ✍️ **My Work**",
    14: "> ✍️ **My Work**",
    16: "> ✍️ **My Work**",
}

new_cells = []
for i, cell in enumerate(nb['cells']):
    if i in markers:
        if cell['cell_type'] == 'markdown':
            cell['source'].insert(0, markers[i] + "\n\n")
            new_cells.append(cell)
        else:
            new_md = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [markers[i]]
            }
            new_cells.append(new_md)
            new_cells.append(cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open('the-platonic-universe/experiment.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
