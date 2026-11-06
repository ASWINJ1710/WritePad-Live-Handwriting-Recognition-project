import torch
from torch.utils.data import DataLoader, random_split
from models.lenet5 import LeNet5
from data_loader import EMNISTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = EMNISTDataset("datasets/emnist_subset")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(
    dataset, [train_size, val_size], generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

model = LeNet5(num_classes=62)

checkpoint = torch.load(
    "model_hub/pre_trained/lenet.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)
model.to(DEVICE)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print("\nPRETRAINED MODEL EVALUATION")
print("---------------------------")
print("Device:", DEVICE)
print("Total validation images:", total)
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
