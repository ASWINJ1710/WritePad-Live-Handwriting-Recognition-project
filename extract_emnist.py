import zipfile
import os

archive = os.path.expanduser("~/Downloads/by_class.zip")
destination = "datasets/emnist_byclass"

os.makedirs(destination, exist_ok=True)

with zipfile.ZipFile(archive, "r") as z:
    files = z.infolist()
    total = len(files)

    for i, file in enumerate(files, 1):
        z.extract(file, destination)

        if i % 1000 == 0 or i == total:
            print(f"Extracted {i}/{total} files")

print("EMNIST extraction completed!")
