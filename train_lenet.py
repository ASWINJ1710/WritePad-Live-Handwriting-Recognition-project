import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from data_loader import EMNISTDataset
from models.lenet5 import LeNet5

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.0001

# Dataset
print("Loading EMNIST dataset...")
dataset = EMNISTDataset("datasets/emnist_subset")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Model
print("\nLoading pretrained LeNet...")
model = LeNet5(num_classes=62)

checkpoint = torch.load(
    "model_hub/pre_trained/lenet.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Save directory
save_dir = Path("model_hub/fine_tuned")
save_dir.mkdir(parents=True, exist_ok=True)

best_accuracy = 0.0

print("\nStarting fine-tuning...")
print("Device:", DEVICE)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Training
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    average_loss = running_loss / len(train_loader)

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Loss: {average_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            save_dir / "lenet_emnist_finetuned.pth"
        )

        print("Best model saved!")

print("\nFine-tuning completed!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Model: model_hub/fine_tuned/lenet_emnist_finetuned.pth")
