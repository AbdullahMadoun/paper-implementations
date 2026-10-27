#!/bin/bash
echo "Starting Colab CPU run for L1=0.01"
/opt/homebrew/bin/colab new -s sae_l1_0_01
/opt/homebrew/bin/colab upload -s sae_l1_0_01 sparse-autoencoders/experiment_l1_0.01.ipynb /content/experiment.ipynb
/opt/homebrew/bin/colab upload -s sae_l1_0_01 colab_runner.py /content/colab_runner.py
/opt/homebrew/bin/colab upload -s sae_l1_0_01 check_done.py /content/check_done.py

/opt/homebrew/bin/colab exec -s sae_l1_0_01 -f colab_runner.py

echo "Waiting for background notebook execution to finish..."
while true; do
    RESULT=$(/opt/homebrew/bin/colab exec -s sae_l1_0_01 -f check_done.py)
    if [[ "$RESULT" == *"YES"* ]]; then
        echo "Execution finished!"
        break
    fi
    sleep 10
done

/opt/homebrew/bin/colab download -s sae_l1_0_01 /content/experiment.ipynb sparse-autoencoders/experiment_l1_0.01.ipynb
# Download the trained weights if they exist
/opt/homebrew/bin/colab download -s sae_l1_0_01 /content/sae_weights.pt sparse-autoencoders/sae_weights_l1_0.01.pt || echo "Weights not found!"
/opt/homebrew/bin/colab stop -s sae_l1_0_01
echo "Finished Colab run for L1=0.01"
