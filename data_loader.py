from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class EMNISTDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root_dir}"
            )

        self.class_folders = sorted(
            folder for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        self.classes = [
            chr(int(folder.name, 16))
            for folder in self.class_folders
        ]

        self.class_to_idx = {
            char: index
            for index, char in enumerate(self.classes)
        }

        self.samples = []

        for folder in self.class_folders:

            label = self.class_to_idx[
                chr(int(folder.name, 16))
            ]

            for image_path in folder.rglob("*.png"):
                self.samples.append((image_path, label))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        print("EMNIST Dataset Loaded!")
        print("Total images:", len(self.samples))
        print("Total classes:", len(self.classes))
        print("Classes:", self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
