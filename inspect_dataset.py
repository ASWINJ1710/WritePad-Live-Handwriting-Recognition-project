from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

classes = {
    "41": "A",
    "43": "C",
    "4d": "M",
    "61": "a",
    "63": "c",
    "71": "q"
}

fig, axes = plt.subplots(6, 6, figsize=(14, 6))

for row, (class_id, character) in enumerate(classes.items()):
    folder = Path("datasets/emnist_subset") / class_id
    images = list(folder.glob("*.png"))[:6]

    for col, image_path in enumerate(images):
        image = Image.open(image_path).convert("L")
        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].set_title(character)
        axes[row, col].axis("off")

plt.suptitle("EMNIST Dataset Inspection")
plt.tight_layout()
plt.show()
