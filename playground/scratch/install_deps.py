import subprocess
import sys
print("Installing datasets...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "scikit-learn", "pandas", "numpy"])
print("Done!")
