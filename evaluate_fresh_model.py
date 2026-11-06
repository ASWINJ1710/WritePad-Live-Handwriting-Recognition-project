import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models.lenet5 import LeNet5

DATA_DIR = "datasets/emnist_subset"
MODEL_PATH = "model_hub/fine_tuned/lenet_emnist_fresh.pth"
CLASS_PATH = "model_hub/fine_tuned/lenet_class_names.pth"

BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading validation dataset...")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Reproduce the same 80/20 split used during training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = torch.load(
    CLASS_PATH,
    map_location="cpu",
    weights_only=False
)

print("Validation images:", len(val_dataset))
print("Classes:", len(class_names))

print("\nLoading trained model...")

model = LeNet5(num_classes=len(class_names)).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

criterion = nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
total = 0

confusion = torch.zeros(
    len(class_names),
    len(class_names),
    dtype=torch.int64
)

print("\nEvaluating model...")

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for actual, predicted in zip(
            labels.cpu(), predictions.cpu()
        ):
            confusion[actual, predicted] += 1

accuracy = 100 * correct / total
average_loss = total_loss / total

print("\n========== EVALUATION RESULTS ==========")
print(f"Validation images: {total}")
print(f"Average loss: {average_loss:.4f}")
print(f"Overall accuracy: {accuracy:.2f}%")

print("\n========== PER-CHARACTER ACCURACY ==========")

for i, character in enumerate(class_names):
    class_total = confusion[i].sum().item()
    class_correct = confusion[i, i].item()

    class_accuracy = (
        100 * class_correct / class_total
        if class_total > 0 else 0
    )

    print(
        f"{character}: {class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )

print("\n========== MOST CONFUSED CHARACTERS ==========")

confusions = []

for actual in range(len(class_names)):
    for predicted in range(len(class_names)):
        if actual != predicted:
            count = confusion[actual, predicted].item()

            if count > 0:
                confusions.append(
                    (count, class_names[actual], class_names[predicted])
                )

confusions.sort(reverse=True)

for count, actual, predicted in confusions[:15]:
    print(f"Actual '{actual}' predicted as '{predicted}': {count} times")

print("\nEvaluation completed!")
