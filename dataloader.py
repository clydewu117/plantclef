import pandas as pd
from torch.utils.data import DataLoader
from dataset import PlantCLEFDataset

def get_label_map(train_csv):
    df = pd.read_csv(train_csv)
    df["species_id"] = df["species_id"].astype(str)
    uniq = sorted(df["species_id"].unique())
    return {sid: i for i, sid in enumerate(uniq)}

def get_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    train_transform,
    eval_transform,
    batch_size=128,
    num_workers=8,
):
    label_map = get_label_map(train_csv)

    train_ds = PlantCLEFDataset(train_csv, transform=train_transform, label_map=label_map)
    val_ds   = PlantCLEFDataset(val_csv,   transform=eval_transform,  label_map=label_map)
    test_ds  = PlantCLEFDataset(test_csv,  transform=eval_transform,  label_map=label_map)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    return train_loader, val_loader, test_loader, label_map
