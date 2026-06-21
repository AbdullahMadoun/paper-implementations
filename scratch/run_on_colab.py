import os
import sys

# The Colab VM needs to install the necessary packages
print("Installing dependencies on Colab...")
os.system("pip install transformers datasets")

print("Executing notebook...")
ret = os.system("jupyter nbconvert --execute --inplace experiment.ipynb")
if ret != 0:
    print(f"Failed with return code {ret}")
    sys.exit(1)
print("Finished execution.")
