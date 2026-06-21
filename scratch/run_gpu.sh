#!/bin/bash
SESSION="sae_gpu_test"
echo "Starting Colab GPU run for L1=0.01"
/opt/homebrew/bin/colab upload -s $SESSION sparse-autoencoders/experiment_l1_0.01.ipynb /content/experiment.ipynb
/opt/homebrew/bin/colab upload -s $SESSION colab_runner.py /content/colab_runner.py
/opt/homebrew/bin/colab upload -s $SESSION check_done.py /content/check_done.py

/opt/homebrew/bin/colab exec -s $SESSION -f colab_runner.py

echo "Waiting for background notebook execution to finish..."
while true; do
    RESULT=$(/opt/homebrew/bin/colab exec -s $SESSION -f check_done.py)
    if [[ "$RESULT" == *"YES"* ]]; then
        echo "Execution finished!"
        break
    fi
    sleep 5
done

/opt/homebrew/bin/colab download -s $SESSION /content/experiment.ipynb sparse-autoencoders/experiment_gpu.ipynb
/opt/homebrew/bin/colab download -s $SESSION /content/sae_weights.pt sparse-autoencoders/sae_weights.pt
/opt/homebrew/bin/colab stop -s $SESSION
echo "Done!"
