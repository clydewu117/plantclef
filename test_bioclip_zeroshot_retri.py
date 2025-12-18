import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
QUERY_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/query.csv"
GALLERY_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/gallery.csv"

BATCH_SIZE = 256
NUM_WORKERS = 8
MAX_QUERIES = 10000  # Limit number of queries to avoid OOM
MAX_GALLERY = 50000  # Limit number of gallery samples to avoid OOM during extraction

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)


def read_split_csv(path: str) -> pd.DataFrame:
    """Read split CSV with comma-separated format"""
    df = pd.read_csv(path)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.strip('"')
    )
    if "species_id" not in df.columns or "image_path" not in df.columns:
        raise KeyError(
            f"Split CSV must have columns image_path,species_id. Got: {list(df.columns)}"
        )
    df["species_id"] = df["species_id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df


def build_label_map(query_df: pd.DataFrame, gallery_df: pd.DataFrame):
    """Build label map from query and gallery species"""
    all_species = sorted(
        set(query_df["species_id"].unique()) | set(gallery_df["species_id"].unique())
    )
    return {sid: i for i, sid in enumerate(all_species)}


class RetrievalImageDataset(Dataset):
    def __init__(
        self, split_df: pd.DataFrame, label_map: dict, preprocess, data_root: str
    ):
        self.df = split_df.reset_index(drop=True)
        self.label_map = label_map
        self.preprocess = preprocess
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["image_path"]
        sid = row["species_id"]

        img_path = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.join(self.data_root, rel_path)
        )
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)

        label = self.label_map[sid]
        return img, label


@torch.no_grad()
def extract_bioclip_features(model, dataloader, device, desc="Extracting features"):
    """Extract image features using BioCLIP"""
    model.eval()

    all_features = []
    all_labels = []

    num_batches = len(dataloader)
    print(f"{desc}...")

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True)

        # Extract image features
        img_features = model.encode_image(imgs)
        img_features = F.normalize(img_features, p=2, dim=1)

        all_features.append(img_features.cpu())
        all_labels.append(labels.cpu())

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(
                f"  Batch [{batch_idx + 1}/{num_batches}] | "
                f"Processed {(batch_idx + 1) * len(imgs)} samples"
            )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def compute_average_precision_cpu(sorted_labels, query_label):
    """Compute Average Precision for a single query (CPU version)"""
    matches = (sorted_labels == query_label).numpy()

    if matches.sum() == 0:
        return 0.0

    precisions = []
    num_relevant = 0

    for i, match in enumerate(matches):
        if match:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)


@torch.no_grad()
def evaluate_retrieval():
    """Evaluate BioCLIP zero-shot retrieval"""
    # Load BioCLIP model
    print("Loading BioCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model = model.to(DEVICE).eval()
    print("BioCLIP model loaded")

    # Load data splits
    print("\nLoading data splits...")
    query_df = read_split_csv(QUERY_CSV)
    gallery_df = read_split_csv(GALLERY_CSV)

    # Build label mapping
    label_map = build_label_map(query_df, gallery_df)
    num_classes = len(label_map)
    print(f"Num test species: {num_classes}")

    # Create datasets
    query_ds = RetrievalImageDataset(query_df, label_map, preprocess, DATA_ROOT)
    gallery_ds = RetrievalImageDataset(gallery_df, label_map, preprocess, DATA_ROOT)

    query_loader = DataLoader(
        query_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print(f"Query set size: {len(query_ds)}")
    print(f"Gallery set size: {len(gallery_ds)}")

    # Extract query features
    print("\n" + "=" * 60)
    print("Step 1/3: Extracting query features")
    print("=" * 60)
    query_features, query_labels = extract_bioclip_features(
        model, query_loader, DEVICE, "Extracting query features"
    )

    # Limit number of queries to avoid OOM
    if query_features.size(0) > MAX_QUERIES:
        print(f"Limiting queries from {query_features.size(0)} to {MAX_QUERIES}")
        query_features = query_features[:MAX_QUERIES]
        query_labels = query_labels[:MAX_QUERIES]

    print(f"✓ Query features extracted: {query_features.size(0)} samples")

    # Extract gallery features
    print("\n" + "=" * 60)
    print("Step 2/3: Extracting gallery features")
    print("=" * 60)
    gallery_features, gallery_labels = extract_bioclip_features(
        model, gallery_loader, DEVICE, "Extracting gallery features"
    )

    # Limit number of gallery samples to avoid OOM
    if gallery_features.size(0) > MAX_GALLERY:
        print(f"Limiting gallery from {gallery_features.size(0)} to {MAX_GALLERY}")
        gallery_features = gallery_features[:MAX_GALLERY]
        gallery_labels = gallery_labels[:MAX_GALLERY]

    print(f"✓ Gallery features extracted: {gallery_features.size(0)} samples")

    # Compute retrieval metrics (memory-efficient version)
    print("\n" + "=" * 60)
    print("Step 3/3: Computing retrieval metrics")
    print("=" * 60)

    n_query = query_features.size(0)
    k_values = [1, 5, 10, 20, 50, 100]
    recall_counts = {k: 0 for k in k_values}
    average_precisions = []

    # Process queries in batches to avoid OOM
    # Use batch size of 100 since we limited total queries and gallery
    query_batch_size = 100
    num_batches = (n_query + query_batch_size - 1) // query_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * query_batch_size
        end_idx = min((batch_idx + 1) * query_batch_size, n_query)

        # Get batch of query features
        query_batch = query_features[start_idx:end_idx]
        query_labels_batch = query_labels[start_idx:end_idx]

        # Compute distances to all gallery samples (on CPU)
        distances = torch.cdist(query_batch, gallery_features, p=2)

        # Get sorted indices for each query
        sorted_indices = distances.argsort(dim=1)

        # Free memory immediately
        del distances
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Compute metrics for this batch
        for i in range(query_batch.size(0)):
            query_label = query_labels_batch[i]

            # Compute Recall@K
            for k in k_values:
                top_k_indices = sorted_indices[i, :k]
                top_k_labels = gallery_labels[top_k_indices]
                if (top_k_labels == query_label).any():
                    recall_counts[k] += 1

            # Compute Average Precision
            sorted_gallery_labels = gallery_labels[sorted_indices[i]]
            ap = compute_average_precision_cpu(sorted_gallery_labels, query_label)
            average_precisions.append(ap)

        # Clean up after each batch
        del query_batch, query_labels_batch, sorted_indices

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Processed {end_idx}/{n_query} queries...")

    # Compute final metrics
    recalls = {f"recall@{k}": recall_counts[k] / n_query for k in k_values}
    mAP = np.mean(average_precisions)

    metrics = {"mAP": mAP, **recalls}

    print("✓ Metrics computed")

    return metrics, query_features.size(0), gallery_features.size(0), num_classes


# Run evaluation
metrics, n_query, n_gallery, num_classes = evaluate_retrieval()

# Print results
print("\n" + "=" * 60)
print("BioCLIP Zero-Shot Retrieval Test Results")
print("=" * 60)
print(f"mAP:           {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
print(f"Recall@1:      {metrics['recall@1']:.4f} ({metrics['recall@1']*100:.2f}%)")
print(f"Recall@5:      {metrics['recall@5']:.4f} ({metrics['recall@5']*100:.2f}%)")
print(f"Recall@10:     {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)")
print(f"Recall@20:     {metrics['recall@20']:.4f} ({metrics['recall@20']*100:.2f}%)")
print(f"Recall@50:     {metrics['recall@50']:.4f} ({metrics['recall@50']*100:.2f}%)")
print(f"Recall@100:    {metrics['recall@100']:.4f} ({metrics['recall@100']*100:.2f}%)")
print("=" * 60)

# Save results to file
result_file = os.path.join(RESULTS_DIR, "bioclip_zeroshot_retrieval_test.txt")

with open(result_file, "w") as f:
    f.write("BioCLIP Zero-Shot Retrieval Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: BioCLIP (hf-hub:imageomics/bioclip)\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Number of Query Samples: {n_query} (limited from full dataset)\n")
    f.write(f"Number of Gallery Samples: {n_gallery} (limited from full dataset)\n")
    f.write(f"Number of Test Species: {num_classes}\n")
    f.write(f"Test Type: Zero-shot (no fine-tuning)\n")
    f.write("=" * 60 + "\n")
    f.write(f"mAP:           {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)\n")
    f.write(
        f"Recall@1:      {metrics['recall@1']:.4f} ({metrics['recall@1']*100:.2f}%)\n"
    )
    f.write(
        f"Recall@5:      {metrics['recall@5']:.4f} ({metrics['recall@5']*100:.2f}%)\n"
    )
    f.write(
        f"Recall@10:     {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)\n"
    )
    f.write(
        f"Recall@20:     {metrics['recall@20']:.4f} ({metrics['recall@20']*100:.2f}%)\n"
    )
    f.write(
        f"Recall@50:     {metrics['recall@50']:.4f} ({metrics['recall@50']*100:.2f}%)\n"
    )
    f.write(
        f"Recall@100:    {metrics['recall@100']:.4f} ({metrics['recall@100']*100:.2f}%)\n"
    )
    f.write("=" * 60 + "\n")
    f.write("\nMetric Descriptions:\n")
    f.write("- mAP: Mean Average Precision across all queries\n")
    f.write(
        "- Recall@K: Percentage of queries where correct species appears in top-K results\n"
    )
    f.write("\nTask Description:\n")
    f.write("This is a cross-species retrieval task where the model is evaluated on\n")
    f.write("plant species that were NOT seen during training. The query and gallery\n")
    f.write("sets contain different species from the training set.\n")
    f.write("\nNote:\n")
    f.write("This is a zero-shot evaluation where BioCLIP was NOT fine-tuned on the\n")
    f.write(
        "training set. Retrieval is performed using pre-trained image embeddings.\n"
    )

print(f"\nResults saved to: {result_file}")
