import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.lenet5 import LeNet5

# ---------------- CONFIGURATION ----------------

DATA_DIR = "datasets/emnist_subset"
SAVE_DIR = "model_hub/fine_tuned"

MODEL_PATH = os.path.join(SAVE_DIR, "lenet_emnist_v2.pth")
CLASS_PATH = os.path.join(SAVE_DIR, "lenet_v2_class_names.pth")

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.RandomAffine(
        degrees=8,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10),
        shear=5,
        fill=255
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------- DATASET ----------------

print("Loading EMNIST subset...")

train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_base = datasets.ImageFolder(DATA_DIR, transform=val_transform)

# Convert hexadecimal folder names into characters
class_names = [chr(int(folder, 16)) for folder in train_base.classes]

print("Total images:", len(train_base))
print("Total classes:", len(class_names))
print("Classes:", class_names)

# Same deterministic 80/20 split as V1
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_base), generator=generator).tolist()

train_size = int(0.8 * len(indices))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

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

# ---------------- MODEL ----------------

print("\nInitializing LeNet-5 V2...")

model = LeNet5(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

best_accuracy = 0.0
early_stop_counter = 0

print("\nStarting V2 training...")
print("Device:", device)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)

# ---------------- TRAINING ----------------

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_accuracy = 100 * train_correct / train_total

    # ---------------- VALIDATION ----------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            val_loss += loss.item() * labels.size(0)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_accuracy)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # ---------------- SAVE BEST MODEL ----------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(class_names, CLASS_PATH)

        print("Best V2 model saved!")

        early_stop_counter = 0

    else:
        early_stop_counter += 1
        print("No improvement:", early_stop_counter, "/", PATIENCE)

    if early_stop_counter >= PATIENCE:
        print("\nEarly stopping triggered!")
        break

print("\n========== V2 TRAINING COMPLETED ==========")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Saved model:", MODEL_PATH)
