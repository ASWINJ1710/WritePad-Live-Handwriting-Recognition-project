import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models.lenet5 import LeNet5
import math

DATA_DIR = "datasets/emnist_subset"
MODEL_PATH = "model_hub/fine_tuned/lenet_emnist_fresh.pth"
CLASS_PATH = "model_hub/fine_tuned/lenet_class_names.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading validation dataset...")

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(
    dataset, [train_size, val_size], generator=generator
)

loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=0
)

class_names = torch.load(
    CLASS_PATH, map_location="cpu", weights_only=False
)

model = LeNet5(num_classes=len(class_names)).to(device)

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

mistakes = []

print("Finding incorrect predictions...")

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]

        for idx in wrong_indices:
            mistakes.append((
                images[idx].cpu().squeeze().numpy(),
                class_names[labels[idx].item()],
                class_names[predictions[idx].item()],
                confidences[idx].item()
            ))

            if len(mistakes) >= 25:
                break

        if len(mistakes) >= 25:
            break

print("Incorrect samples collected:", len(mistakes))

if not mistakes:
    print("No mistakes found!")
    raise SystemExit

columns = 5
rows = math.ceil(len(mistakes) / columns)

fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
axes = axes.flatten()

for i, (image, actual, predicted, confidence) in enumerate(mistakes):
    # Undo normalization for display
    image = image * 0.5 + 0.5

    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(
        f"Actual: {actual} | Pred: {predicted}\n"
        f"Confidence: {confidence:.1%}",
        fontsize=9
    )
    axes[i].axis("off")

for i in range(len(mistakes), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("model_hub/fine_tuned/misclassified_samples.png", dpi=150)
plt.show()

print("Saved: model_hub/fine_tuned/misclassified_samples.png")
