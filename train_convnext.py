import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

from dataloader import get_dataloaders

# =========================
# 配置区（一眼就能看懂）
# =========================
TRAIN_CSV = "splits/train.csv"
VAL_CSV   = "splits/val.csv"
TEST_CSV  = "splits/test.csv"

BATCH_SIZE = 128          # A100 推荐
EPOCHS = 10                # baseline 跑 2–3 个就够
LR = 1e-3                 # 大 batch 稍大 LR
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 50
NUM_WORKERS = 8

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
std  = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.Resize(236),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

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
)

num_classes = len(label_map)
print("Num classes:", num_classes)

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
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        if step == 1 or step % PRINT_EVERY == 0:
            batch_acc = preds.eq(labels).float().mean().item()
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3
                if DEVICE == "cuda" else 0.0
            )
            print(
                f"[Epoch {epoch}] "
                f"Step [{step}/{num_steps}] "
                f"Loss: {loss.item():.4f} "
                f"BatchAcc: {batch_acc:.4f} "
                f"GPU_mem: {mem:.2f}GB"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

    return total_loss / total, correct / total

# =========================
# Training Loop
# =========================
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, epoch)
    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"\nEpoch [{epoch}/{EPOCHS}] DONE | "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}\n"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "convnext_baseline_best.pth")
        print(f"✓ Saved new best model (val acc = {val_acc:.4f})\n")

print("Training finished.")
print("Best Val Acc:", best_val_acc)
