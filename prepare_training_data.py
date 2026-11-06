from pathlib import Path
from collections import defaultdict
import random

SOURCE = Path("datasets/emnist_byclass/by_class")
OUTPUT = Path("datasets/emnist_subset")
SAMPLES_PER_CLASS = 1000
SEED = 42

random.seed(SEED)

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

OUTPUT.mkdir(parents=True, exist_ok=True)

for class_folder in sorted(SOURCE.iterdir()):
    if not class_folder.is_dir():
        continue

    images = list(class_folder.rglob("*.png"))
    selected = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

    destination = OUTPUT / class_folder.name
    destination.mkdir(parents=True, exist_ok=True)

    for image in selected:
        target = destination / image.name
        if not target.exists():
            import shutil
            shutil.copy2(image, target)

    print(f"{class_folder.name}: {len(selected)} images")

print("\nEMNIST subset preparation completed!")
print("Output:", OUTPUT)
