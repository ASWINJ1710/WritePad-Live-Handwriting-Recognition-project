from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

dataset = Path("datasets/emnist_subset")

classes = ["41", "42", "30", "61", "39"]

fig, axes = plt.subplots(1, len(classes), figsize=(12, 3))

for ax, class_name in zip(axes, classes):
    folder = dataset / class_name
    images = list(folder.glob("*.png"))

    if not images:
        ax.set_title(f"{class_name}: No image")
        ax.axis("off")
        continue

    image = Image.open(images[0]).convert("L")

    ax.imshow(image, cmap="gray")
    ax.set_title(f"{class_name} = {chr(int(class_name, 16))}")
    ax.axis("off")

plt.tight_layout()
plt.show()
