import os
log_path = "/content/colab_mknn.log"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    print("=== LOG OUTPUT ===")
    # Print the last 40 lines
    lines = content.splitlines()
    print("\n".join(lines[-40:]))
    print("==================")
else:
    print("Log file not found.")
