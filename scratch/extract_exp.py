import tarfile

with tarfile.open("pu.tar.gz", "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith(".py") and "experiment" in member.name:
            print("=========================================")
            print("Found file:", member.name)
            print("=========================================")
            f = tar.extractfile(member)
            if f:
                print(f.read().decode())
