import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data_loader import EMNISTDataset
from models.lenet5 import LeNet5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = EMNISTDataset("datasets/emnist_subset")

# Select 128 images
subset = Subset(dataset, list(range(128)))

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# Create a fresh model (no pretrained weights)
model = LeNet5(num_classes=62).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nStarting overfitting sanity test...")
print("Device:", device)
print("Images:", len(subset))

# Train repeatedly on the same 128 images
for epoch in range(30):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/30 | "
        f"Loss: {total_loss / len(loader):.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

print("\nSanity test completed!")
