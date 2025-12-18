import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

from dataloader_retrieval import get_retrieval_dataloaders
from retrieval_metrics import extract_features
from tqdm import tqdm

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/train_val.csv"
QUERY_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/query.csv"
GALLERY_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/gallery.csv"

BATCH_SIZE = 256
NUM_WORKERS = 8
EMBEDDING_DIM = 512
MAX_QUERIES = 10000  # Limit number of queries to avoid OOM
MAX_GALLERY = 50000  # Limit number of gallery samples to avoid OOM during extraction

CHECKPOINT_PATH = "checkpoints/convnext_retrieval_best.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
mean = weights.transforms().mean
std = weights.transforms().std

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Load query and gallery dataloaders
from dataloader_retrieval import get_query_gallery_loaders

query_loader, gallery_loader, label_map = get_query_gallery_loaders(
    QUERY_CSV,
    GALLERY_CSV,
    transform=eval_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    data_root=DATA_ROOT,
)

num_classes = len(label_map)
print("Num classes (test species):", num_classes)

print("Loading ConvNeXt Tiny model...")
model = convnext_tiny(weights=weights)
in_dim = model.classifier[2].in_features

# Recreate model architecture
backbone = nn.Sequential(*list(model.children())[:-1])
head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),
    nn.Linear(in_dim, EMBEDDING_DIM),
)

backbone = backbone.to(DEVICE)
head = head.to(DEVICE)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
backbone.load_state_dict(checkpoint["backbone"])
head.load_state_dict(checkpoint["head"])
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")


@torch.no_grad()
def extract_features_limited(backbone, head, dataloader, device, max_samples=None):
    """Extract features with optional limit on number of samples"""
    backbone.eval()
    head.eval()

    all_features = []
    all_labels = []
    total_samples = 0

    for imgs, labels in tqdm(dataloader, desc="Extracting features"):
        imgs = imgs.to(device, non_blocking=True)

        feats = backbone(imgs)
        embeddings = head(feats)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_features.append(embeddings.cpu())
        all_labels.append(labels.cpu())

        total_samples += imgs.size(0)

        # Stop early if we've reached the limit
        if max_samples is not None and total_samples >= max_samples:
            break

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Trim to exact max_samples if specified
    if max_samples is not None and features.size(0) > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]

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
    backbone.eval()
    head.eval()

    # Extract query features (with limit to avoid OOM)
    print("\n" + "="*60)
    print("Step 1/3: Extracting query features...")
    print("="*60)
    query_features, query_labels = extract_features_limited(
        backbone, head, query_loader, DEVICE, max_samples=MAX_QUERIES
    )
    print(f"✓ Query features extracted: {query_features.size(0)} samples")

    # Extract gallery features (with limit to avoid OOM)
    print("\n" + "="*60)
    print("Step 2/3: Extracting gallery features...")
    print("="*60)
    gallery_features, gallery_labels = extract_features_limited(
        backbone, head, gallery_loader, DEVICE, max_samples=MAX_GALLERY
    )
    print(f"✓ Gallery features extracted: {gallery_features.size(0)} samples")

    # Compute retrieval metrics (memory-efficient version)
    print("\n" + "="*60)
    print("Step 3/3: Computing retrieval metrics...")
    print("="*60)

    n_query = query_features.size(0)
    k_values = [1, 5, 10, 20, 50, 100]
    recall_counts = {k: 0 for k in k_values}
    average_precisions = []

    # Process queries in batches to avoid OOM
    # Use batch size of 100 since we limited total queries
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

    metrics = {
        "mAP": mAP,
        **recalls
    }

    print("✓ Metrics computed")

    return metrics, query_features.size(0), gallery_features.size(0)


# Run evaluation
metrics, n_query, n_gallery = evaluate_retrieval()

# Print results
print("\n" + "=" * 60)
print("ConvNeXt Tiny Retrieval Test Results")
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
result_file = os.path.join(RESULTS_DIR, "convnext_retrieval_test.txt")

with open(result_file, "w") as f:
    f.write("ConvNeXt Tiny Retrieval Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Embedding Dimension: {EMBEDDING_DIM}\n")
    f.write(f"Number of Query Samples: {n_query} (limited from full dataset)\n")
    f.write(f"Number of Gallery Samples: {n_gallery} (limited from full dataset)\n")
    f.write(f"Number of Test Species: {num_classes}\n")
    f.write("=" * 60 + "\n")
    f.write(f"mAP:           {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)\n")
    f.write(f"Recall@1:      {metrics['recall@1']:.4f} ({metrics['recall@1']*100:.2f}%)\n")
    f.write(f"Recall@5:      {metrics['recall@5']:.4f} ({metrics['recall@5']*100:.2f}%)\n")
    f.write(f"Recall@10:     {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)\n")
    f.write(f"Recall@20:     {metrics['recall@20']:.4f} ({metrics['recall@20']*100:.2f}%)\n")
    f.write(f"Recall@50:     {metrics['recall@50']:.4f} ({metrics['recall@50']*100:.2f}%)\n")
    f.write(f"Recall@100:    {metrics['recall@100']:.4f} ({metrics['recall@100']*100:.2f}%)\n")
    f.write("=" * 60 + "\n")
    f.write("\nMetric Descriptions:\n")
    f.write("- mAP: Mean Average Precision across all queries\n")
    f.write("- Recall@K: Percentage of queries where correct species appears in top-K results\n")
    f.write("\nTask Description:\n")
    f.write("This is a cross-species retrieval task where the model is evaluated on\n")
    f.write("plant species that were NOT seen during training. The query and gallery\n")
    f.write("sets contain different species from the training set.\n")

print(f"\nResults saved to: {result_file}")
