import tarfile

# Read downloaded file from:
# http://deepyeti.ucsd.edu/jmcauley/datasets/librarything/lthing_data.tar.gz
with tarfile.open("lthing_data.tar.gz") as tar:
    print("Files in tar archive:")
    tar.list()

    with tar.extractfile("lthing_data/reviews.json") as file:
        count = 0
        for line in file:
            print(line)
            count += 1
            if count > 3:
                break
