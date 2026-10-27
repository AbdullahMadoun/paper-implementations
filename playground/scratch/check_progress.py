import os
print(os.system("ps aux | grep python"))
print("Done exists:", os.path.exists('/content/done.txt'))
print("Error exists:", os.path.exists('/content/error.txt'))
if os.path.exists('/content/error.txt'):
    with open('/content/error.txt', 'r') as f:
        print("Error content:")
        print(f.read())
