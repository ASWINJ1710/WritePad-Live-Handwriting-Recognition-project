import torch
from torch.utils.data import DataLoader, random_split
from models.lenet5 import LeNet5
from data_loader import EMNISTDataset

device = torch.device("cuda")

dataset = EMNISTDataset("datasets/emnist_subset")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(
    dataset, [train_size, val_size], generator=generator
)

loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = LeNet5(num_classes=62)
checkpoint = torch.load(
    "model_hub/pre_trained/lenet.pth",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(checkpoint, strict=True)
model.to(device)
model.eval()

results = {"normal": 0, "inverted": 0}
total = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        normal = images * 0.5 + 0.5
        inverted = 1.0 - normal

        normal_output = model(normal)
        inverted_output = model(inverted)

        results["normal"] += (
            normal_output.argmax(1) == labels
        ).sum().item()

        results["inverted"] += (
            inverted_output.argmax(1) == labels
        ).sum().item()

        total += labels.size(0)

print("\nPOLARITY TEST")
print("Normal accuracy:", f"{100 * results['normal'] / total:.2f}%")
print("Inverted accuracy:", f"{100 * results['inverted'] / total:.2f}%")
