import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from collections import defaultdict

class PlantCLEFRetrievalDataset(Dataset):
    """
    Dataset for retrieval task with better sampling support
    """
    def __init__(self, csv_file, transform=None, label_map=None, data_root=None):
        """
        csv_file: train.csv / val.csv / query.csv / gallery.csv
        transform: torchvision transforms
        label_map: species_id -> label mapping (must be shared across splits)
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

        # Build index for PK sampling
        self.label_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            label = int(row["label"])
            self.label_to_indices[label].append(idx)

        self.labels = self.df["label"].values
        self.unique_labels = sorted(self.label_to_indices.keys())

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

    def get_labels(self):
        """Return all labels for sampler"""
        return self.labels

    def get_label_to_indices(self):
        """Return mapping from label to sample indices for PK sampler"""
        return self.label_to_indices


class PKSampler(torch.utils.data.Sampler):
    """
    PK Sampler: Sample P classes, and K samples per class in each batch

    This ensures each batch has multiple samples per class, which is crucial
    for triplet mining in retrieval tasks.
    """
    def __init__(self, dataset, p_classes, k_samples, shuffle=True):
        """
        Args:
            dataset: PlantCLEFRetrievalDataset instance
            p_classes: number of classes per batch
            k_samples: number of samples per class
            shuffle: whether to shuffle classes and samples
        """
        self.dataset = dataset
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.shuffle = shuffle

        self.label_to_indices = dataset.get_label_to_indices()
        self.labels = list(self.label_to_indices.keys())

        # Filter out classes with fewer than k_samples
        self.labels = [
            label for label in self.labels
            if len(self.label_to_indices[label]) >= k_samples
        ]

        if len(self.labels) < p_classes:
            raise ValueError(
                f"Not enough classes with >= {k_samples} samples. "
                f"Found {len(self.labels)}, need at least {p_classes}"
            )

        self.num_classes = len(self.labels)
        self.batch_size = p_classes * k_samples

        # Calculate number of batches
        self.num_batches = len(self.labels) // p_classes

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.labels)

        for batch_idx in range(self.num_batches):
            batch_indices = []

            # Select P classes for this batch
            start_idx = batch_idx * self.p_classes
            end_idx = start_idx + self.p_classes
            batch_classes = self.labels[start_idx:end_idx]

            for label in batch_classes:
                # Get all indices for this class
                indices = self.label_to_indices[label].copy()

                if self.shuffle:
                    np.random.shuffle(indices)

                # Sample K samples from this class
                selected = indices[:self.k_samples]
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return self.num_batches


class BatchHardTripletSampler(torch.utils.data.Sampler):
    """
    Standard random sampler but ensures diverse batches for triplet mining
    by sampling uniformly across classes
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.label_to_indices = dataset.get_label_to_indices()
        self.labels = list(self.label_to_indices.keys())
        self.num_classes = len(self.labels)

    def __iter__(self):
        # Create a balanced sample pool
        all_indices = []

        for label in self.labels:
            indices = self.label_to_indices[label]
            all_indices.extend(indices)

        if self.shuffle:
            np.random.shuffle(all_indices)

        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        return total_samples // self.batch_size
