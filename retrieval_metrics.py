import torch
import numpy as np
from tqdm import tqdm


def compute_distance_matrix(query_features, gallery_features):
    """
    Compute pairwise distance matrix between query and gallery features

    Args:
        query_features: (N_query, D) tensor
        gallery_features: (N_gallery, D) tensor

    Returns:
        distance_matrix: (N_query, N_gallery) tensor
    """
    return torch.cdist(query_features, gallery_features, p=2)


def compute_recall_at_k(distance_matrix, query_labels, gallery_labels, k_values=[1, 5, 10, 20]):
    """
    Compute Recall@K for retrieval

    Args:
        distance_matrix: (N_query, N_gallery) distance matrix
        query_labels: (N_query,) labels for queries
        gallery_labels: (N_gallery,) labels for gallery
        k_values: list of K values to compute

    Returns:
        dict of recall@k values
    """
    n_query = distance_matrix.size(0)

    # Get sorted indices (ascending distance)
    sorted_indices = distance_matrix.argsort(dim=1)  # (N_query, N_gallery)

    recalls = {}

    for k in k_values:
        correct = 0
        for i in range(n_query):
            query_label = query_labels[i]
            # Get top-k gallery indices
            top_k_indices = sorted_indices[i, :k]
            top_k_labels = gallery_labels[top_k_indices]

            # Check if any of top-k matches query label
            if (top_k_labels == query_label).any():
                correct += 1

        recalls[f"recall@{k}"] = correct / n_query

    return recalls


def compute_average_precision(sorted_labels, query_label):
    """
    Compute Average Precision for a single query

    Args:
        sorted_labels: gallery labels sorted by distance to query
        query_label: label of the query

    Returns:
        average_precision: float
    """
    # Find positions where label matches
    matches = (sorted_labels == query_label).cpu().numpy()

    if matches.sum() == 0:
        return 0.0

    # Compute precision at each relevant position
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


def compute_map(distance_matrix, query_labels, gallery_labels):
    """
    Compute Mean Average Precision (mAP)

    Args:
        distance_matrix: (N_query, N_gallery) distance matrix
        query_labels: (N_query,) labels for queries
        gallery_labels: (N_gallery,) labels for gallery

    Returns:
        mAP: mean average precision
    """
    n_query = distance_matrix.size(0)

    # Get sorted indices
    sorted_indices = distance_matrix.argsort(dim=1)

    average_precisions = []

    for i in range(n_query):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[sorted_indices[i]]

        ap = compute_average_precision(sorted_gallery_labels, query_label)
        average_precisions.append(ap)

    return np.mean(average_precisions)


def compute_retrieval_metrics(query_features, query_labels, gallery_features, gallery_labels, k_values=[1, 5, 10, 20]):
    """
    Compute all retrieval metrics

    Args:
        query_features: (N_query, D) tensor
        query_labels: (N_query,) tensor
        gallery_features: (N_gallery, D) tensor
        gallery_labels: (N_gallery,) tensor
        k_values: list of K values for Recall@K

    Returns:
        dict of metrics
    """
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(query_features, gallery_features)

    # Compute Recall@K
    recalls = compute_recall_at_k(distance_matrix, query_labels, gallery_labels, k_values)

    # Compute mAP
    mAP = compute_map(distance_matrix, query_labels, gallery_labels)

    metrics = {
        "mAP": mAP,
        **recalls
    }

    return metrics


@torch.no_grad()
def extract_features(model_backbone, model_head, dataloader, device):
    """
    Extract features from a dataloader

    Args:
        model_backbone: backbone model
        model_head: embedding head model
        dataloader: data loader
        device: device to use

    Returns:
        features: (N, D) tensor of features
        labels: (N,) tensor of labels
    """
    model_backbone.eval()
    model_head.eval()

    all_features = []
    all_labels = []

    for imgs, labels in tqdm(dataloader, desc="Extracting features"):
        imgs = imgs.to(device, non_blocking=True)

        # Forward pass
        if hasattr(model_backbone, 'forward_features'):
            # DINOv2 style
            out = model_backbone.forward_features(imgs)
            feats = out["x_norm_clstoken"]
        else:
            # ConvNeXt style
            feats = model_backbone(imgs)

        embeddings = model_head(feats)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_features.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def evaluate_retrieval(backbone, head, val_loader, device, batch_size=1000):
    """
    Evaluate retrieval performance on validation set (same-species retrieval)

    Memory-efficient version that processes queries in batches to avoid OOM.

    Args:
        backbone: backbone model
        head: embedding head
        val_loader: validation dataloader
        device: device
        batch_size: number of queries to process at once (default: 1000)

    Returns:
        dict of metrics
    """
    # Extract all features
    features, labels = extract_features(backbone, head, val_loader, device)

    # Move features to CPU to save GPU memory
    features = features.cpu()
    labels = labels.cpu()

    n = features.size(0)
    print(f"Evaluating retrieval on {n} samples (batch_size={batch_size})...")

    # Initialize counters for metrics
    k_values = [1, 5, 10, 20]
    recall_counts = {k: 0 for k in k_values}
    average_precisions = []

    # Process queries in batches
    num_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n)

        # Get batch of query features
        query_feats = features[start_idx:end_idx]  # (batch, D)
        query_labels = labels[start_idx:end_idx]  # (batch,)

        # Compute distances to all gallery samples (excluding self)
        # (batch, n)
        distances = torch.cdist(query_feats, features, p=2)

        # Set self-matches to infinity
        for i in range(query_feats.size(0)):
            global_idx = start_idx + i
            distances[i, global_idx] = float('inf')

        # Get sorted indices for each query
        sorted_indices = distances.argsort(dim=1)  # (batch, n)

        # Compute Recall@K for this batch
        for i in range(query_feats.size(0)):
            query_label = query_labels[i]

            for k in k_values:
                top_k_indices = sorted_indices[i, :k]
                top_k_labels = labels[top_k_indices]
                if (top_k_labels == query_label).any():
                    recall_counts[k] += 1

            # Compute Average Precision for this query
            sorted_gallery_labels = labels[sorted_indices[i]]
            ap = compute_average_precision(sorted_gallery_labels, query_label)
            average_precisions.append(ap)

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Processed {end_idx}/{n} queries...")

    # Compute final metrics
    recalls = {f"recall@{k}": recall_counts[k] / n for k in k_values}
    mAP = np.mean(average_precisions)

    metrics = {
        "mAP": mAP,
        **recalls
    }

    return metrics
