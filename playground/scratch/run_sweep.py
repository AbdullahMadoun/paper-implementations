import json
import subprocess
import os
import time

l1_values = [0.01, 0.05]
base_notebook_path = "sparse-autoencoders/experiment.ipynb"

with open(base_notebook_path, "r") as f:
    nb_base = json.load(f)

for l1 in l1_values:
    # Parameterize notebook
    nb_copy = json.loads(json.dumps(nb_base)) 
    for cell in nb_copy["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            for i, line in enumerate(source):
                if "lmb = 0.02" in line or "lmb=0.02" in line:
                    source[i] = line.replace("0.02", str(l1))
    
    nb_filename = f"sparse-autoencoders/experiment_l1_{l1}.ipynb"
    with open(nb_filename, "w") as f:
        json.dump(nb_copy, f, indent=1)
        
    session_name = f"sae_l1_{str(l1).replace('.', '_')}"
    
    script_content = f"""#!/bin/bash
echo "Starting Colab CPU run for L1={l1}"
/opt/homebrew/bin/colab new -s {session_name}
/opt/homebrew/bin/colab upload -s {session_name} {nb_filename} /content/experiment.ipynb
/opt/homebrew/bin/colab upload -s {session_name} colab_runner.py /content/colab_runner.py
/opt/homebrew/bin/colab upload -s {session_name} check_done.py /content/check_done.py

/opt/homebrew/bin/colab exec -s {session_name} -f colab_runner.py

echo "Waiting for background notebook execution to finish..."
while true; do
    RESULT=$(/opt/homebrew/bin/colab exec -s {session_name} -f check_done.py)
    if [[ "$RESULT" == *"YES"* ]]; then
        echo "Execution finished!"
        break
    fi
    sleep 10
done

/opt/homebrew/bin/colab download -s {session_name} /content/experiment.ipynb {nb_filename}
# Download the trained weights if they exist
/opt/homebrew/bin/colab download -s {session_name} /content/sae_weights.pt sparse-autoencoders/sae_weights_l1_{l1}.pt || echo "Weights not found!"
/opt/homebrew/bin/colab stop -s {session_name}
echo "Finished Colab run for L1={l1}"
"""
    script_name = f"run_{session_name}.sh"
    with open(script_name, "w") as f:
        f.write(script_content)
        
    os.chmod(script_name, 0o755)
    
    print(f"Running sweep for L1={l1} sequentially on CPU...")
    subprocess.run(["./" + script_name])
    os.remove(script_name)

print("All sweep jobs completed.")
