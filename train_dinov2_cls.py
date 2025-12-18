import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast
from torchvision import transforms

import wandb

from dataloader import get_dataloaders

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/splits/val.csv"
TEST_CSV = "/fs/scratch/PAS2099/plantclef/splits/test.csv"

BATCH_SIZE = 512
EPOCHS = 10

FREEZE_EPOCHS = 1
BASE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 1

NUM_WORKERS = 8
TOP_K = 5
PRINT_EVERY = 20

DINOV2_MODEL = "dinov2_vitb14"

USE_WANDB = True
WANDB_PROJECT = "plantclef"
WANDB_RUN_NAME = "dinov2_vitb14_class"

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

if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": DINOV2_MODEL,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS,
            "base_lr": BASE_LR,
            "head_lr": HEAD_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "top_k": TOP_K,
            "num_workers": NUM_WORKERS,
            "amp": "bf16",
            "gpu": "A100-80GB",
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

head = nn.Linear(feature_dim, num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = AdamW(
    [
        {"params": backbone.parameters(), "lr": BASE_LR},
        {"params": head.parameters(), "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * WARMUP_EPOCHS


def lr_factor(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / float(denom)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def get_cls_features(x: torch.Tensor) -> torch.Tensor:
    """Extract CLS token features from DINOv2 backbone"""
    out = backbone.forward_features(x)
    if "x_norm_clstoken" not in out:
        raise KeyError(
            f"forward_features keys = {list(out.keys())}, expected 'x_norm_clstoken'"
        )
    return out["x_norm_clstoken"]


def freeze_backbone(do_freeze: bool):
    for p in backbone.parameters():
        p.requires_grad = not do_freeze


def train_one_epoch(epoch: int):
    backbone.train()
    head.train()

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for step, (imgs, labels) in enumerate(train_loader, start=1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            feats = get_cls_features(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if any(p.requires_grad for p in backbone.parameters()):
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        correct1 += preds.eq(labels).sum().item()

        topk = logits.topk(TOP_K, dim=1).indices
        correct5 += topk.eq(labels[:, None]).any(dim=1).sum().item()

        if step == 1 or step % PRINT_EVERY == 0:
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
            )
            lr_backbone = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
            batch_top1 = preds.eq(labels).float().mean().item()
            batch_top5 = topk.eq(labels[:, None]).any(dim=1).float().mean().item()

            print(
                f"[Epoch {epoch}] Step [{step}/{steps_per_epoch}] "
                f"Loss {loss.item():.4f} "
                f"Top1 {batch_top1:.4f} "
                f"Top5 {batch_top5:.4f} "
                f"LR(bb) {lr_backbone:.2e} LR(head) {lr_head:.2e} "
                f"GPU {mem:.2f}GB"
            )

            if USE_WANDB:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/batch_top1": batch_top1,
                        "train/batch_top5": batch_top5,
                        "train/lr_backbone": lr_backbone,
                        "train/lr_head": lr_head,
                        "train/gpu_mem_gb": mem,
                        "train/step": (epoch - 1) * steps_per_epoch + step,
                    }
                )

    return total_loss / total, correct1 / total, correct5 / total


@torch.no_grad()
def evaluate():
    backbone.eval()
    head.eval()

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            feats = get_cls_features(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        correct1 += preds.eq(labels).sum().item()

        topk = logits.topk(TOP_K, dim=1).indices
        correct5 += topk.eq(labels[:, None]).any(dim=1).sum().item()

    return total_loss / total, correct1 / total, correct5 / total


best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    if epoch <= FREEZE_EPOCHS:
        freeze_backbone(True)
        print(f"[Epoch {epoch}] Backbone frozen (linear probe stage)")
    else:
        freeze_backbone(False)
        print(f"[Epoch {epoch}] Backbone unfrozen (finetune stage)")

    train_loss, train_acc, train_top5 = train_one_epoch(epoch)
    val_loss, val_acc, val_top5 = evaluate()

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] DONE | "
        f"Train loss {train_loss:.4f}, top1 {train_acc:.4f}, top5 {train_top5:.4f} | "
        f"Val loss {val_loss:.4f}, top1 {val_acc:.4f}, top5 {val_top5:.4f}\n"
    )

    if USE_WANDB:
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_top1": train_acc,
                "train/epoch_top5": train_top5,
                "val/loss": val_loss,
                "val/top1": val_acc,
                "val/top5": val_top5,
            }
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(CHECKPOINT_DIR, "dinov2_class_best.pth")
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "label_map": label_map,
            },
            ckpt_path,
        )
        print(f"✓ Saved best model: {ckpt_path} (val top1 = {val_acc:.4f})")

print("Training finished.")
print("Best Val Top-1 Acc:", best_val_acc)

if USE_WANDB:
    wandb.finish()
