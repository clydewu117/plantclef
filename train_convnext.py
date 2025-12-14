import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

import wandb
import os

from dataloader import get_dataloaders

# =========================
# 配置区（一眼就能看懂）
# =========================
DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/splits/val.csv"
TEST_CSV = "/fs/scratch/PAS2099/plantclef/splits/test.csv"

BATCH_SIZE = 128  # A100 推荐
EPOCHS = 10  # baseline 跑 2–3 个就够
LR = 1e-3  # 大 batch 稍大 LR
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 50
NUM_WORKERS = 8
TOP_K = 5  # 计算 Top-5 accuracy

# Wandb 配置
USE_WANDB = True
WANDB_PROJECT = "plantclef"
WANDB_RUN_NAME = "convnext_tiny_baseline"

# Checkpoint 配置
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A100 / Ampere 优化
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Using device:", DEVICE)

# =========================
# Transforms（ConvNeXt 专属）
# =========================
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose(
    [
        transforms.Resize(236),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# =========================
# Data
# =========================
train_loader, val_loader, _, label_map = get_dataloaders(
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    train_transform=train_transform,
    eval_transform=eval_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    data_root=DATA_ROOT,
)

num_classes = len(label_map)
print("Num classes:", num_classes)

# =========================
# Wandb 初始化
# =========================
if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "model": "convnext_tiny",
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "label_smoothing": 0.1,
            "top_k": TOP_K,
        },
    )

# =========================
# Model
# =========================
model = convnext_tiny(weights=weights)
in_dim = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_dim, num_classes)
model = model.to(DEVICE)

# =========================
# Optimizer / Loss
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler(enabled=(DEVICE == "cuda"))


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    num_steps = len(loader)

    for step, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs

        # Top-1 accuracy
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5_preds = logits.topk(TOP_K, dim=1, largest=True, sorted=True)
        correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

        total += bs

        if step == 1 or step % PRINT_EVERY == 0:
            batch_acc = preds.eq(labels).float().mean().item()
            batch_top5_acc = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).any(dim=1).float().mean().item()
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
            )
            print(
                f"[Epoch {epoch}] "
                f"Step [{step}/{num_steps}] "
                f"Loss: {loss.item():.4f} "
                f"Top1: {batch_acc:.4f} "
                f"Top5: {batch_top5_acc:.4f} "
                f"GPU_mem: {mem:.2f}GB"
            )

            # Log to wandb (batch-level metrics)
            if USE_WANDB:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_top1_acc": batch_acc,
                    "train/batch_top5_acc": batch_top5_acc,
                    "train/gpu_memory_gb": mem,
                    "train/step": (epoch - 1) * num_steps + step,
                })

    return total_loss / total, correct / total, correct_top5 / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs

        # Top-1 accuracy
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5_preds = logits.topk(TOP_K, dim=1, largest=True, sorted=True)
        correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

        total += bs

    return total_loss / total, correct / total, correct_top5 / total


# =========================
# Training Loop
# =========================
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_top5 = train_one_epoch(model, train_loader, epoch)
    val_loss, val_acc, val_top5 = evaluate(model, val_loader)

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] DONE | "
        f"Train loss: {train_loss:.4f}, top1: {train_acc:.4f}, top5: {train_top5:.4f} | "
        f"Val loss: {val_loss:.4f}, top1: {val_acc:.4f}, top5: {val_top5:.4f}\n"
    )

    # Log epoch-level metrics to wandb
    if USE_WANDB:
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_top1_acc": train_acc,
            "train/epoch_top5_acc": train_top5,
            "val/loss": val_loss,
            "val/top1_acc": val_acc,
            "val/top5_acc": val_top5,
        })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "convnext_baseline_best.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✓ Saved new best model to {checkpoint_path} (val top1 acc = {val_acc:.4f})\n")

        # Log best model to wandb
        if USE_WANDB:
            wandb.log({"val/best_top1_acc": best_val_acc})

print("Training finished.")
print("Best Val Top-1 Acc:", best_val_acc)

# Finish wandb run
if USE_WANDB:
    wandb.finish()
