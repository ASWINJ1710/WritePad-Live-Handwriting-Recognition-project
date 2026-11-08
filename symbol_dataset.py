from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class HASYDataset(Dataset):

    def __init__(self, root_dir, csv_file, class_to_idx=None):

        self.root_dir = Path(root_dir)
        self.csv_file = Path(csv_file)

        self.data = pd.read_csv(self.csv_file)

        if class_to_idx is None:
            labels = sorted(self.data["latex"].unique())
            self.class_to_idx = {
                label: index for index, label in enumerate(labels)
            }
        else:
            self.class_to_idx = class_to_idx

        self.data = self.data[
            self.data["latex"].isin(self.class_to_idx)
        ].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (self.csv_file.parent / row["path"]).resolve()

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        label = self.class_to_idx[row["latex"]]

        return image, torch.tensor(label, dtype=torch.long)
