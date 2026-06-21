import os
import subprocess

# Find the PID of the process running colab_mknn.py
try:
    output = subprocess.check_output(["pgrep", "-f", "colab_mknn.py"]).decode().strip()
    pids = output.split()
    for pid in pids:
        print(f"Killing process {pid}...")
        os.system(f"kill -9 {pid}")
except Exception as e:
    print("No running colab_mknn.py process found or error:", e)
