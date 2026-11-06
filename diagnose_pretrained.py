import torch
from torch.utils.data import DataLoader, random_split
from models.lenet5 import LeNet5
from data_loader import EMNISTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

results = {}

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        images = images * 0.5 + 0.5

        for rotation in range(4):
            rotated = torch.rot90(images, rotation, dims=(-2, -1))

            for flip in [False, True]:
                transformed = torch.flip(rotated, dims=(-1,)) if flip else rotated

                for norm in ["zero_one", "minus_one_one"]:
                    inputs = transformed if norm == "zero_one" else transformed * 2 - 1

                    predictions = model(inputs).argmax(dim=1)

                    key = (rotation, flip, norm)
                    if key not in results:
                        results[key] = 0

                    results[key] += (predictions == labels).sum().item()

print("\nPRETRAINED MODEL DIAGNOSTICS")
print("----------------------------")

for key, correct in sorted(results.items(), key=lambda x: x[1], reverse=True)[:8]:
    rotation, flip, norm = key
    print(f"Rotation: {rotation * 90} degrees | Flip: {flip} | Normalization: {norm} | Accuracy: {100 * correct / len(val_dataset):.2f}%")
