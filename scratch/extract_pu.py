import urllib.request
import json
import tarfile
import os

url = "https://pypi.org/pypi/platonic-universe/json"
req = urllib.request.urlopen(url)
data = json.loads(req.read().decode())
version = data['info']['version']
urls = data['releases'][version]

sdist_url = None
for item in urls:
    if item['packagetype'] == 'sdist':
        sdist_url = item['url']
        break

if sdist_url:
    print("Downloading:", sdist_url)
    urllib.request.urlretrieve(sdist_url, "pu.tar.gz")
    with tarfile.open("pu.tar.gz", "r:gz") as tar:
        print("Files in archive:")
        for member in tar.getmembers():
            print("  ", member.name)
            if member.name.endswith(".py"):
                print("--- START OF Python File:", member.name, "---")
                f = tar.extractfile(member)
                if f:
                    try:
                        print(f.read().decode())
                    except Exception as e:
                        print("Error reading:", e)
                print("--- END OF Python File ---")
else:
    print("sdist not found")
