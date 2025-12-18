import os
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

from dataloader import get_dataloaders

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/splits/val.csv"
TEST_CSV = "/fs/scratch/PAS2099/plantclef/splits/test.csv"

BATCH_SIZE = 256
NUM_WORKERS = 8
TOP_K = 5

CHECKPOINT_PATH = "checkpoints/convnext_class_best.pth"
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

_, _, test_loader, label_map = get_dataloaders(
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    train_transform=eval_transform,  # Not used
    eval_transform=eval_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    data_root=DATA_ROOT,
)

num_classes = len(label_map)
print("Num classes:", num_classes)

print("Loading ConvNeXt Tiny model...")
model = convnext_tiny(weights=weights)
in_dim = model.classifier[2].in_features

# Recreate model architecture
backbone = nn.Sequential(*list(model.children())[:-1])
head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),
    nn.LayerNorm(in_dim),
    nn.Linear(in_dim, num_classes),
)

backbone = backbone.to(DEVICE)
head = head.to(DEVICE)

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
backbone.load_state_dict(checkpoint["backbone"])
head.load_state_dict(checkpoint["head"])
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate():
    backbone.eval()
    head.eval()

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    print("Evaluating on test set...")
    num_batches = len(test_loader)
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            feats = backbone(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        correct1 += preds.eq(labels).sum().item()

        topk = logits.topk(TOP_K, dim=1).indices
        correct5 += topk.eq(labels[:, None]).any(dim=1).sum().item()

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            current_top1 = correct1 / total
            current_top5 = correct5 / total
            print(f"  Batch [{batch_idx + 1}/{num_batches}] | "
                  f"Samples: {total} | "
                  f"Top-1: {current_top1*100:.2f}% | "
                  f"Top-5: {current_top5*100:.2f}%")

    avg_loss = total_loss / total
    top1_acc = correct1 / total
    top5_acc = correct5 / total

    return avg_loss, top1_acc, top5_acc


# Run evaluation
test_loss, test_top1, test_top5 = evaluate()

# Print results
print("\n" + "=" * 60)
print("ConvNeXt Tiny Classification Test Results")
print("=" * 60)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Top-1:    {test_top1:.4f} ({test_top1*100:.2f}%)")
print(f"Test Top-5:    {test_top5:.4f} ({test_top5*100:.2f}%)")
print("=" * 60)

# Save results to file
result_file = os.path.join(RESULTS_DIR, "convnext_class_test.txt")

with open(result_file, "w") as f:
    f.write("ConvNeXt Tiny Classification Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Loss:     {test_loss:.4f}\n")
    f.write(f"Test Top-1:    {test_top1:.4f} ({test_top1*100:.2f}%)\n")
    f.write(f"Test Top-5:    {test_top5:.4f} ({test_top5*100:.2f}%)\n")
    f.write("=" * 60 + "\n")

print(f"\nResults saved to: {result_file}")
