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
        label_map: 提供 species_id -> new_label 的字典
                    (train & val & test 必须共享同一份)
        data_root: 数据根目录，如果CSV中是相对路径则需要指定
        """

        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.data_root = data_root

        # species_id -> string
        self.df["species_id"] = self.df["species_id"].astype(str)

        # 如果外部没有传 label_map，则内部生成（只适用于 train）
        if label_map is None:
            unique_species = sorted(self.df["species_id"].unique())
            self.label_map = {sid: idx for idx, sid in enumerate(unique_species)}
        else:
            self.label_map = label_map

        # 把 label 变成训练用的 0...C-1 索引
        self.df["label"] = self.df["species_id"].apply(lambda x: self.label_map[x])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        # 如果指定了 data_root，则拼接路径
        if self.data_root is not None:
            img_path = os.path.join(self.data_root, img_path)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
