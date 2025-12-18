import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
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

# Learning rate scheduler
WARMUP_EPOCHS = 2  # Warmup for first 2 epochs
USE_COSINE_DECAY = True  # Use cosine annealing after warmup

EMBEDDING_DIM = 512

# ArcFace hyperparameters
ARCFACE_SCALE = 30.0  # s: scaling factor for logits
ARCFACE_MARGIN = 0.5  # m: angular margin in radians (about 28.6 degrees)

NUM_WORKERS = 8
PRINT_EVERY = 20

DINOV2_MODEL = "dinov2_vitb14"

USE_WANDB = True
WANDB_PROJECT = "plantclef"
WANDB_RUN_NAME = "dinov2_vitb14_retrieval"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
            "model": DINOV2_MODEL,
            "task": "retrieval",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS,
            "base_lr": BASE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "embedding_dim": EMBEDDING_DIM,
            "arcface_scale": ARCFACE_SCALE,
            "arcface_margin": ARCFACE_MARGIN,
            "num_workers": NUM_WORKERS,
            "amp": "bf16",
            "loss": "ArcFace",
        },
    )

print(f"Loading DINOv2 backbone: {DINOV2_MODEL}")
backbone = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL)
backbone = backbone.to(DEVICE)

feature_dim_map = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
feature_dim = feature_dim_map[DINOV2_MODEL]

# Embedding head for retrieval
head = nn.Linear(feature_dim, EMBEDDING_DIM).to(DEVICE)


# ArcFace Loss Implementation
class ArcFaceLayer(nn.Module):
    """ArcFace: Additive Angular Margin Loss for Deep Face Recognition"""

    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight matrix for classification (learnable class centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, in_features) - should be L2 normalized
        labels: (batch_size,) - ground truth class labels
        """
        # Normalize weight vectors (class centers on unit hypersphere)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: dot product of normalized embeddings and weights
        cosine = F.linear(embeddings, normalized_weight)  # (batch, num_classes)
        cosine = cosine.clamp(-1.0, 1.0)  # Numerical stability

        # Get the cosine of (theta + margin) for ground truth classes
        theta = torch.acos(cosine)
        target_theta = theta.clone()

        # Add angular margin to ground truth class
        one_hot = F.one_hot(labels, self.out_features).float()
        target_theta += one_hot * self.margin

        # Convert back to cosine
        target_cosine = torch.cos(target_theta)

        # Scale logits
        logits = target_cosine * self.scale

        return logits


# ArcFace layer and cross-entropy loss
arcface = ArcFaceLayer(
    in_features=EMBEDDING_DIM, out_features=num_classes, scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    [
        {"params": backbone.parameters(), "lr": BASE_LR},
        {"params": head.parameters(), "lr": HEAD_LR},
        {"params": arcface.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch


def get_lr_lambda(current_step):
    """Learning rate schedule: warmup + cosine decay"""
    # Warmup phase
    if current_step < warmup_steps:
        return current_step / warmup_steps

    # Cosine decay phase
    if USE_COSINE_DECAY:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Constant after warmup
    return 1.0


scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda)


def get_patch_features(x: torch.Tensor) -> torch.Tensor:
    """Extract patch token mean pooling features from DINOv2 backbone (better for retrieval)"""
    out = backbone.forward_features(x)
    if "x_norm_patchtokens" not in out:
        raise KeyError(
            f"forward_features keys = {list(out.keys())}, expected 'x_norm_patchtokens'"
        )
    # Mean pooling over all patch tokens (excluding CLS)
    return out["x_norm_patchtokens"].mean(dim=1)


def freeze_backbone(do_freeze: bool):
    for p in backbone.parameters():
        p.requires_grad = not do_freeze


def train_one_epoch(epoch: int):
    backbone.train()
    head.train()
    arcface.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (imgs, labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            feats = get_patch_features(imgs)
            embeddings = head(feats)
            # L2 normalize embeddings for ArcFace
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # ArcFace forward: returns logits
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), 1.0)
        if any(p.requires_grad for p in backbone.parameters()):
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)

        optimizer.step()
        scheduler.step()  # Update learning rate

        # Calculate accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

        bs = imgs.size(0)
        total_loss += loss.item() * bs

        if step == 1 or step % PRINT_EVERY == 0:
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
            )
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
            batch_acc = correct / bs

            print(
                f"[Epoch {epoch}] Step [{step}/{steps_per_epoch}] "
                f"Loss {loss.item():.4f} Acc {batch_acc:.4f} "
                f"LR(bb) {lr_backbone:.2e} LR(head) {lr_head:.2e} "
                f"GPU {mem:.2f}GB"
            )

            if USE_WANDB:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/batch_acc": batch_acc,
                        "train/lr_backbone": lr_backbone,
                        "train/lr_head": lr_head,
                        "train/gpu_mem_gb": mem,
                        "train/step": (epoch - 1) * steps_per_epoch + step,
                    }
                )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate():
    backbone.eval()
    head.eval()
    arcface.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            feats = get_patch_features(imgs)
            embeddings = head(feats)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


best_mAP = 0.0

for epoch in range(1, EPOCHS + 1):
    if epoch <= FREEZE_EPOCHS:
        freeze_backbone(True)
        print(f"[Epoch {epoch}] Backbone frozen (linear probe stage)")
    else:
        freeze_backbone(False)
        print(f"[Epoch {epoch}] Backbone unfrozen (finetune stage)")

    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate()

    # Compute retrieval metrics (mAP, Recall@K)
    print("Computing retrieval metrics...")
    retrieval_metrics = evaluate_retrieval(backbone, head, val_loader, DEVICE)

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] DONE | "
        f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"Val loss {val_loss:.4f} acc {val_acc:.4f}\n"
        f"Retrieval: mAP={retrieval_metrics['mAP']:.4f}, "
        f"R@1={retrieval_metrics['recall@1']:.4f}, "
        f"R@5={retrieval_metrics['recall@5']:.4f}, "
        f"R@10={retrieval_metrics['recall@10']:.4f}\n"
    )

    if USE_WANDB:
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/mAP": retrieval_metrics["mAP"],
                "val/recall@1": retrieval_metrics["recall@1"],
                "val/recall@5": retrieval_metrics["recall@5"],
                "val/recall@10": retrieval_metrics["recall@10"],
                "val/recall@20": retrieval_metrics["recall@20"],
            }
        )

    # Save best model based on mAP (not loss)
    current_mAP = retrieval_metrics["mAP"]
    if current_mAP > best_mAP:
        best_mAP = current_mAP
        ckpt_path = os.path.join(CHECKPOINT_DIR, "dinov2_retrieval_best.pth")
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "mAP": current_mAP,
                "label_map": label_map,
                "embedding_dim": EMBEDDING_DIM,
                "arcface_scale": ARCFACE_SCALE,
                "arcface_margin": ARCFACE_MARGIN,
            },
            ckpt_path,
        )
        print(f"✓ Saved best model: {ckpt_path} (mAP = {current_mAP:.4f})")

print("Training finished.")
print("Best mAP:", best_mAP)

if USE_WANDB:
    wandb.finish()
