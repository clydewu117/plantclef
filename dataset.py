import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class PlantCLEFDataset(Dataset):
    def __init__(self, csv_file, transform=None, label_map=None, data_root=None):
        """
        csv_file: train.csv / val.csv / test.csv
        transform: torchvision transforms
        label_map: species_id -> label mapping (must be shared across train/val/test)
        data_root: root directory for data if CSV contains relative paths
        """

        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_root = data_root

        self.df["species_id"] = self.df["species_id"].astype(str)

        if label_map is None:
            unique_species = sorted(self.df["species_id"].unique())
            self.label_map = {sid: idx for idx, sid in enumerate(unique_species)}
        else:
            self.label_map = label_map

        self.df["label"] = self.df["species_id"].apply(lambda x: self.label_map[x])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        if self.data_root is not None:
            img_path = os.path.join(self.data_root, img_path)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
