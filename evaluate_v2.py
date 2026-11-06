import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models.lenet5 import LeNet5

DATA_DIR = "datasets/emnist_subset"
MODEL_PATH = "model_hub/fine_tuned/lenet_emnist_v2.pth"
CLASS_PATH = "model_hub/fine_tuned/lenet_v2_class_names.pth"

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
    dataset,
    [train_size, val_size],
    generator=generator
)

loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

class_names = torch.load(
    CLASS_PATH,
    map_location="cpu",
    weights_only=False
)

model = LeNet5(num_classes=len(class_names)).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

confusion = torch.zeros(
    len(class_names),
    len(class_names),
    dtype=torch.int64
)

correct = 0
total = 0

print("Evaluating V2 model...")

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for actual, predicted in zip(
            labels.cpu(), predictions.cpu()
        ):
            confusion[actual, predicted] += 1

print("\n========== V2 RESULTS ==========")
print(f"Overall accuracy: {100 * correct / total:.2f}%")

print("\n========== PER-CHARACTER ACCURACY ==========")

for i, character in enumerate(class_names):
    count = confusion[i].sum().item()
    right = confusion[i, i].item()
    accuracy = 100 * right / count if count else 0

    print(f"{character}: {accuracy:.2f}% ({right}/{count})")

print("\n========== MOST CONFUSED CHARACTERS ==========")

confusions = []

for actual in range(len(class_names)):
    for predicted in range(len(class_names)):
        if actual != predicted:
            count = confusion[actual, predicted].item()
            if count:
                confusions.append((
                    count,
                    class_names[actual],
                    class_names[predicted]
                ))

for count, actual, predicted in sorted(confusions, reverse=True)[:15]:
    print(f"'{actual}' predicted as '{predicted}': {count} times")

print("\nV2 evaluation completed!")
