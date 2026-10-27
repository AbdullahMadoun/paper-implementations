import json

with open('the-platonic-universe/experiment.ipynb') as f:
    nb = json.load(f)

old = nb['cells']
new_cells = []

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

new_cells.append(md_cell("# The Platonic Universe: Do Foundation Models See the Same Sky?\n\nThis notebook reproduces the core representational alignment experiments from the paper, verifying the Platonic Representation Hypothesis for astronomical foundation models."))

new_cells.append(md_cell("## 1. Setup & Imports\n\n> 🤖 **AI-Assisted**"))
new_cells.append(old[1])

new_cells.append(md_cell("## 2. Data Loading\n\n> 🤖 **AI-Assisted**\n\nLoad the crossmatched dataset containing pre-computed embeddings for HSC and JWST."))
new_cells.append(old[3])
new_cells.append(md_cell("> ✍️ **My Work**"))
new_cells.append(old[5])

mknn_note = ''.join(old[6]['source'])
new_cells.append(md_cell(f"## 3. Mutual K-Nearest Neighbors (MKNN)\n\n{mknn_note}"))
new_cells.append(old[7])

new_cells.append(md_cell("## 4. Intramodal Alignment\n\n> ✍️ **My Work**\n\nFirst, we compute the representational alignment between different model sizes within the same modality."))
new_cells.append(old[9])
new_cells.append(md_cell("> ✍️ **My Work**"))
new_cells.append(old[11])
new_cells.append(md_cell("> ✍️ **My Work**"))
new_cells.append(old[13])

hsc_note = ''.join(old[14]['source']).replace('> ✍️ **My Work**', '').strip()
new_cells.append(md_cell(f"> ✍️ **My Work**\n\n{hsc_note}"))
new_cells.append(old[16])

new_cells.append(md_cell("## 5. Cross-Modal Alignment\n\n> ✍️ **My Work**\n\nNext, we compare the embeddings across different modalities (HSC vs JWST)."))
new_cells.append(old[20])
new_cells.append(md_cell("> 🤖 **AI-Assisted**\n\nModel parameter counts (in millions) for scaling analysis."))
new_cells.append(old[22])

new_cells.append(md_cell("> ✍️ **My Work**\n\nVisualizing the scaling behavior of cross-modal alignment."))
new_cells.append(old[24])
new_cells.append(old[26])

conc = ''.join(old[28]['source']).replace('> ✍️ **My Work**', '').strip()
new_cells.append(md_cell(f"## 6. Conclusion\n\n> ✍️ **My Work**\n\n{conc}"))

nb['cells'] = new_cells

with open('the-platonic-universe/experiment.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
