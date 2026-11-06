import torch
from torch.utils.data import DataLoader, random_split
from data_loader import EMNISTDataset

DATASET_PATH = "datasets/emnist_subset"
BATCH_SIZE = 128

dataset = EMNISTDataset(DATASET_PATH)

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
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Dataset preparation successful!")
print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes:", len(dataset.classes))
print("Class order:", dataset.classes)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
