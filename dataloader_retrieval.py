import pandas as pd
from torch.utils.data import DataLoader
from dataset_retrieval import PlantCLEFRetrievalDataset, PKSampler, BatchHardTripletSampler


def get_label_map(train_csv):
    """Build label map from training CSV"""
    df = pd.read_csv(train_csv)
    df["species_id"] = df["species_id"].astype(str)
    uniq = sorted(df["species_id"].unique())
    return {sid: i for i, sid in enumerate(uniq)}


def get_retrieval_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    train_transform,
    eval_transform,
    batch_size=128,
    num_workers=8,
    data_root=None,
    use_pk_sampler=True,
    p_classes=16,
    k_samples=4,
):
    """
    Create dataloaders for retrieval task

    Args:
        train_csv: path to training CSV
        val_csv: path to validation CSV
        test_csv: path to test CSV (not used in retrieval)
        train_transform: transforms for training
        eval_transform: transforms for evaluation
        batch_size: batch size (only used if use_pk_sampler=False)
        num_workers: number of dataloader workers
        data_root: root directory for images
        use_pk_sampler: whether to use PK sampler for training
        p_classes: number of classes per batch (for PK sampler)
        k_samples: number of samples per class (for PK sampler)

    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    label_map = get_label_map(train_csv)

    train_ds = PlantCLEFRetrievalDataset(
        train_csv, transform=train_transform, label_map=label_map, data_root=data_root
    )
    val_ds = PlantCLEFRetrievalDataset(
        val_csv, transform=eval_transform, label_map=label_map, data_root=data_root
    )
    test_ds = PlantCLEFRetrievalDataset(
        test_csv, transform=eval_transform, label_map=label_map, data_root=data_root
    )

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    if use_pk_sampler:
        # Use PK sampler for better triplet mining
        train_sampler = PKSampler(
            train_ds, p_classes=p_classes, k_samples=k_samples, shuffle=True
        )
        effective_batch_size = p_classes * k_samples
        print(f"Using PK Sampler: P={p_classes}, K={k_samples}, effective batch size={effective_batch_size}")

        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            **common
        )
    else:
        # Use standard random sampling with batch-hard mining
        print(f"Using standard random sampling with batch size={batch_size}")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **common
        )

    # Validation and test loaders use standard sampling
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, label_map


def get_query_gallery_loaders(
    query_csv,
    gallery_csv,
    transform,
    batch_size=256,
    num_workers=8,
    data_root=None,
    label_map=None,
):
    """
    Create dataloaders for retrieval evaluation (query and gallery sets)

    Args:
        query_csv: path to query CSV (unseen species)
        gallery_csv: path to gallery CSV (unseen species)
        transform: transforms for evaluation
        batch_size: batch size
        num_workers: number of dataloader workers
        data_root: root directory for images
        label_map: optional label map (can be None for unseen species)

    Returns:
        query_loader, gallery_loader, query_label_map, gallery_label_map
    """
    # Build label maps for query and gallery
    if label_map is None:
        # Create a unified label map for unseen species
        query_df = pd.read_csv(query_csv)
        gallery_df = pd.read_csv(gallery_csv)
        query_df["species_id"] = query_df["species_id"].astype(str)
        gallery_df["species_id"] = gallery_df["species_id"].astype(str)

        all_species = sorted(
            set(query_df["species_id"].unique()) | set(gallery_df["species_id"].unique())
        )
        unified_label_map = {sid: i for i, sid in enumerate(all_species)}
    else:
        unified_label_map = label_map

    query_ds = PlantCLEFRetrievalDataset(
        query_csv, transform=transform, label_map=unified_label_map, data_root=data_root
    )
    gallery_ds = PlantCLEFRetrievalDataset(
        gallery_csv, transform=transform, label_map=unified_label_map, data_root=data_root
    )

    common = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    query_loader = DataLoader(query_ds, **common)
    gallery_loader = DataLoader(gallery_ds, **common)

    return query_loader, gallery_loader, unified_label_map
