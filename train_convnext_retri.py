import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

import wandb

from dataloader_retrieval import get_retrieval_dataloaders
from retrieval_metrics import evaluate_retrieval

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/train_val.csv"
TEST_CSV = "/fs/scratch/PAS2099/plantclef/re_splits/train.csv"  # Not used for retrieval

BATCH_SIZE = 512
EPOCHS = 10

FREEZE_EPOCHS = 1
BASE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 512
MARGIN = 0.5  # For triplet loss

NUM_WORKERS = 8
PRINT_EVERY = 20

USE_WANDB = True
WANDB_PROJECT = "plantclef"
WANDB_RUN_NAME = "convnext_tiny_retrieval"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_loader, val_loader, _, label_map = get_retrieval_dataloaders(
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    train_transform=train_transform,
    eval_transform=eval_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    data_root=DATA_ROOT,
    use_pk_sampler=True,
    p_classes=32,
    k_samples=16,
)

num_classes = len(label_map)
print("Num classes (train species):", num_classes)

if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": "convnext_tiny",
            "task": "retrieval",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS,
            "base_lr": BASE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "embedding_dim": EMBEDDING_DIM,
            "margin": MARGIN,
            "num_workers": NUM_WORKERS,
            "amp": "bf16",
            "loss": "TripletMarginLoss",
        },
    )

print("Loading ConvNeXt Tiny model...")
model = convnext_tiny(weights=weights)
in_dim = model.classifier[2].in_features

# Split into backbone and embedding head
backbone = nn.Sequential(*list(model.children())[:-1])
head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),
    nn.Linear(in_dim, EMBEDDING_DIM),
)

backbone = backbone.to(DEVICE)
head = head.to(DEVICE)

# Triplet loss for retrieval learning
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

optimizer = AdamW(
    [
        {"params": backbone.parameters(), "lr": BASE_LR},
        {"params": head.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = len(train_loader)


def freeze_backbone(do_freeze: bool):
    for p in backbone.parameters():
        p.requires_grad = not do_freeze


def mine_triplets(embeddings, labels):
    """
    Online triplet mining: for each anchor, find hardest positive and hardest negative
    """
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    triplets = []

    for i in range(len(labels)):
        anchor_label = labels[i]

        # Find hardest positive (same class, farthest)
        pos_mask = (labels == anchor_label) & (
            torch.arange(len(labels), device=labels.device) != i
        )
        if pos_mask.sum() == 0:
            continue
        pos_distances = distance_matrix[i].clone()
        pos_distances[~pos_mask] = -float("inf")
        hardest_pos_idx = pos_distances.argmax()

        # Find hardest negative (different class, closest)
        neg_mask = labels != anchor_label
        if neg_mask.sum() == 0:
            continue
        neg_distances = distance_matrix[i].clone()
        neg_distances[~neg_mask] = float("inf")
        hardest_neg_idx = neg_distances.argmin()

        triplets.append((i, hardest_pos_idx.item(), hardest_neg_idx.item()))

    return triplets


def train_one_epoch(epoch: int):
    backbone.train()
    head.train()

    total_loss = 0.0
    total = 0
    num_triplets = 0

    for step, (imgs, labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            feats = backbone(imgs)
            embeddings = head(feats)
            # L2 normalize embeddings for better retrieval
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Mine hard triplets from batch
        triplets = mine_triplets(embeddings, labels)

        if len(triplets) == 0:
            continue

        # Extract anchor, positive, negative embeddings
        anchors = torch.stack([embeddings[t[0]] for t in triplets])
        positives = torch.stack([embeddings[t[1]] for t in triplets])
        negatives = torch.stack([embeddings[t[2]] for t in triplets])

        loss = criterion(anchors, positives, negatives)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if any(p.requires_grad for p in backbone.parameters()):
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)

        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * len(triplets)
        total += len(triplets)
        num_triplets += len(triplets)

        if step == 1 or step % PRINT_EVERY == 0:
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
            )
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]

            print(
                f"[Epoch {epoch}] Step [{step}/{steps_per_epoch}] "
                f"Loss {loss.item():.4f} "
                f"Triplets {len(triplets)}/{bs} "
                f"LR(bb) {lr_backbone:.2e} LR(head) {lr_head:.2e} "
                f"GPU {mem:.2f}GB"
            )

            if USE_WANDB:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/num_triplets": len(triplets),
                        "train/lr_backbone": lr_backbone,
                        "train/lr_head": lr_head,
                        "train/gpu_mem_gb": mem,
                        "train/step": (epoch - 1) * steps_per_epoch + step,
                    }
                )

    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, num_triplets


@torch.no_grad()
def evaluate():
    backbone.eval()
    head.eval()

    total_loss = 0.0
    total = 0
    num_triplets = 0

    # Collect distances for additional metrics
    all_pos_dists = []
    all_neg_dists = []

    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            feats = backbone(imgs)
            embeddings = head(feats)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        triplets = mine_triplets(embeddings, labels)

        if len(triplets) == 0:
            continue

        anchors = torch.stack([embeddings[t[0]] for t in triplets])
        positives = torch.stack([embeddings[t[1]] for t in triplets])
        negatives = torch.stack([embeddings[t[2]] for t in triplets])

        loss = criterion(anchors, positives, negatives)

        # Compute distances for metrics
        pos_dists = torch.norm(anchors - positives, p=2, dim=1)
        neg_dists = torch.norm(anchors - negatives, p=2, dim=1)

        all_pos_dists.append(pos_dists)
        all_neg_dists.append(neg_dists)

        total_loss += loss.item() * len(triplets)
        total += len(triplets)
        num_triplets += len(triplets)

    avg_loss = total_loss / total if total > 0 else 0.0

    # Compute distance statistics
    if len(all_pos_dists) > 0:
        all_pos_dists = torch.cat(all_pos_dists)
        all_neg_dists = torch.cat(all_neg_dists)
        avg_pos_dist = all_pos_dists.mean().item()
        avg_neg_dist = all_neg_dists.mean().item()
        avg_margin = avg_neg_dist - avg_pos_dist
    else:
        avg_pos_dist = 0.0
        avg_neg_dist = 0.0
        avg_margin = 0.0

    return avg_loss, num_triplets, avg_pos_dist, avg_neg_dist, avg_margin


best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    if epoch <= FREEZE_EPOCHS:
        freeze_backbone(True)
        print(f"[Epoch {epoch}] Backbone frozen (linear probe stage)")
    else:
        freeze_backbone(False)
        print(f"[Epoch {epoch}] Backbone unfrozen (finetune stage)")

    train_loss, train_triplets = train_one_epoch(epoch)
    val_loss, val_triplets, val_pos_dist, val_neg_dist, val_margin = evaluate()

    # Compute retrieval metrics (mAP, Recall@K)
    print("Computing retrieval metrics...")
    retrieval_metrics = evaluate_retrieval(backbone, head, val_loader, DEVICE)

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] DONE | "
        f"Train loss {train_loss:.4f} (triplets={train_triplets}) | "
        f"Val loss {val_loss:.4f} (triplets={val_triplets})\n"
        f"Retrieval: mAP={retrieval_metrics['mAP']:.4f}, "
        f"R@1={retrieval_metrics['recall@1']:.4f}, "
        f"R@5={retrieval_metrics['recall@5']:.4f}, "
        f"R@10={retrieval_metrics['recall@10']:.4f}\n"
        f"Distances: pos={val_pos_dist:.4f}, neg={val_neg_dist:.4f}, margin={val_margin:.4f}\n"
    )

    if USE_WANDB:
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_triplets": train_triplets,
                "val/loss": val_loss,
                "val/triplets": val_triplets,
                "val/pos_distance": val_pos_dist,
                "val/neg_distance": val_neg_dist,
                "val/margin": val_margin,
                "val/mAP": retrieval_metrics["mAP"],
                "val/recall@1": retrieval_metrics["recall@1"],
                "val/recall@5": retrieval_metrics["recall@5"],
                "val/recall@10": retrieval_metrics["recall@10"],
                "val/recall@20": retrieval_metrics["recall@20"],
            }
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt_path = os.path.join(CHECKPOINT_DIR, "convnext_retrieval_best.pth")
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "label_map": label_map,
                "embedding_dim": EMBEDDING_DIM,
            },
            ckpt_path,
        )
        print(f"✓ Saved best model: {ckpt_path} (val loss = {val_loss:.4f})")

print("Training finished.")
print("Best Val Loss:", best_val_loss)

if USE_WANDB:
    wandb.finish()
