from data_loader import EMNISTDataset

dataset = EMNISTDataset(
    "datasets/emnist_byclass/by_class"
)

print("\nTesting first image...")

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label number:", label.item())
print("Character:", dataset.classes[label.item()])
print("Total images:", len(dataset))
