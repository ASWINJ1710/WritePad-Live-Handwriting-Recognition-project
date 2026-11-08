from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from symbol_dataset import HASYDataset
from models.symbol_cnn import SymbolCNN


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path("datasets/hasyv2")
TRAIN_CSV = ROOT / "classification-task/fold-1/train.csv"

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.10
SEED = 42

MODEL_DIR = Path("model_hub/fine_tuned")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hasyv2_symbol_cnn.pth"

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("       HASYv2 SYMBOL MODEL TRAINING")
print("=" * 55)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading HASYv2 dataset...")

dataset = HASYDataset(
    root_dir=ROOT,
    csv_file=TRAIN_CSV
)

num_classes = len(dataset.class_to_idx)

print("Total images:", len(dataset))
print("Number of classes:", num_classes)


# ==========================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ==========================================

print("\nPreparing training and validation split...")

indices_by_class = {}

for index, label in enumerate(dataset.data["latex"]):
    indices_by_class.setdefault(label, []).append(index)

train_indices = []
val_indices = []

rng = random.Random(SEED)

for label, indices in indices_by_class.items():

    rng.shuffle(indices)

    if len(indices) >= 2:
        val_count = max(1, round(len(indices) * VALIDATION_RATIO))
        val_count = min(val_count, len(indices) - 1)
    else:
        val_count = 0

    val_indices.extend(indices[:val_count])
    train_indices.extend(indices[val_count:])

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

print("Training samples:", len(train_subset))
print("Validation samples:", len(val_subset))

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ==========================================
# INITIALIZE MODEL
# ==========================================

print("\nInitializing CNN...")

model = SymbolCNN(num_classes=num_classes).to(device)

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

best_val_accuracy = 0.0

classes = [
    label
    for label, index in sorted(
        dataset.class_to_idx.items(),
        key=lambda item: item[1]
    )
]


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_accuracy = 100.0 * train_correct / train_total


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_accuracy = 100.0 * val_correct / val_total

    scheduler.step(val_accuracy)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": dataset.class_to_idx,
                "classes": classes,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            },
            MODEL_PATH
        )

        print("Best model saved!")


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print("Model saved at:", MODEL_PATH)
