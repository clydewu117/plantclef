import os
import re
import pandas as pd
import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_ROOT = "/fs/scratch/PAS2099/plantclef"
TRAIN_CSV = "/fs/scratch/PAS2099/plantclef/splits/train.csv"
VAL_CSV = "/fs/scratch/PAS2099/plantclef/splits/val.csv"
TEST_CSV = "/fs/scratch/PAS2099/plantclef/splits/test.csv"
META_CSV = "/fs/scratch/PAS2099/plantclef/full_metadata.csv"

BATCH_SIZE = 256
NUM_WORKERS = 8
TOP_K = 5

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


def read_meta_csv(path: str) -> pd.DataFrame:
    """Read metadata CSV with semicolon-separated format"""
    df = pd.read_csv(path, sep=";", quotechar='"', engine="python")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.strip('"')
    )
    if "species_id" not in df.columns:
        raise KeyError(
            f"Metadata CSV missing 'species_id'. Columns: {list(df.columns)}"
        )
    df["species_id"] = df["species_id"].astype(str)
    return df


def normalize_species_name(s: str) -> str:
    """Normalize species names for text prompts"""
    s = re.sub(r"\s+", " ", str(s)).strip()
    parts = s.split(" ")
    if len(parts) >= 3 and re.fullmatch(r"[A-Za-z]\.", parts[-1]):
        s = " ".join(parts[:-1])
    return s


def build_label_map_from_train_split(train_split_df: pd.DataFrame):
    uniq = sorted(train_split_df["species_id"].unique())
    return {sid: i for i, sid in enumerate(uniq)}


def build_sid_to_species(meta_df: pd.DataFrame):
    """Build mapping from species_id to species name"""
    if "species" in meta_df.columns:
        name_col = "species"
    elif "genus" in meta_df.columns:
        name_col = "genus"
    elif "family" in meta_df.columns:
        name_col = "family"
    else:
        raise KeyError(
            f"Metadata has none of ['species','genus','family']. Columns: {list(meta_df.columns)}"
        )

    sid_to_name = (
        meta_df.dropna(subset=[name_col])
        .drop_duplicates(subset=["species_id"])
        .set_index("species_id")[name_col]
        .astype(str)
        .to_dict()
    )
    sid_to_name = {
        sid: normalize_species_name(name) for sid, name in sid_to_name.items()
    }
    return sid_to_name, name_col


class SplitImageDataset(Dataset):
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
def build_text_features(model, tokenizer, classnames, templates):
    """Build text features for all classes by averaging across templates"""
    print("\nBuilding text features for all classes...")
    feats = []
    for i, cname in enumerate(classnames):
        prompts = [t.format(cname) for t in templates]
        tok = tokenizer(prompts).to(DEVICE)
        tf = model.encode_text(tok)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        tf = tf.mean(dim=0)
        tf = tf / tf.norm()
        feats.append(tf)

        # Print progress
        if (i + 1) % 100 == 0 or (i + 1) == len(classnames):
            print(f"  Encoded {i + 1}/{len(classnames)} classes")

    return torch.stack(feats, dim=0)


@torch.no_grad()
def evaluate():
    """Evaluate BioCLIP zero-shot on test set"""
    # Load BioCLIP model
    print("Loading BioCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.to(DEVICE).eval()
    print("BioCLIP model loaded")

    # Load data splits
    print("\nLoading data splits...")
    train_split = read_split_csv(TRAIN_CSV)
    test_split = read_split_csv(TEST_CSV)
    meta = read_meta_csv(META_CSV)

    # Build label mapping from training set
    label_map = build_label_map_from_train_split(train_split)
    idx_to_sid = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"Num classes: {num_classes}")

    # Build species name mapping
    sid_to_species, used_col = build_sid_to_species(meta)
    print(f"Using metadata name column: {used_col}")

    # Check for missing species
    missing = [sid for sid in label_map.keys() if sid not in sid_to_species]
    if len(missing) > 0:
        print(
            f"WARNING: {len(missing)}/{num_classes} species_id not found in metadata '{used_col}'."
        )
        print(f"         Using species_id string as name for missing ones.")

    # Build class names
    classnames = []
    for i in range(num_classes):
        sid = idx_to_sid[i]
        name = sid_to_species.get(sid, sid)
        classnames.append(name)

    print("\nExample class mappings:")
    for i in range(min(5, num_classes)):
        print(f"  idx={i} sid={idx_to_sid[i]} name='{classnames[i]}'")

    # Build text prompts
    templates = [
        "a photo of {}",
        "a photo of the plant {}",
        "a close-up photo of {}",
    ]

    text_features = build_text_features(model, tokenizer, classnames, templates)

    # Create test dataloader
    print("\nPreparing test dataloader...")
    test_ds = SplitImageDataset(test_split, label_map, preprocess, DATA_ROOT)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    print(f"Test set size: {len(test_ds)}")

    # Evaluate
    correct1 = 0
    correct5 = 0
    total = 0

    logit_scale = (
        model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    )

    print("\nEvaluating on test set...")
    num_batches = len(test_loader)

    for batch_idx, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # Encode images
        img_f = model.encode_image(imgs)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        # Compute logits
        logits = logit_scale * (img_f @ text_features.t())

        # Top-1 accuracy
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        # Top-K accuracy
        topk = logits.topk(TOP_K, dim=1).indices
        correct5 += topk.eq(labels[:, None]).any(dim=1).sum().item()

        total += labels.size(0)

        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            current_top1 = correct1 / total
            current_top5 = correct5 / total
            print(
                f"  Batch [{batch_idx + 1}/{num_batches}] | "
                f"Samples: {total} | "
                f"Top-1: {current_top1*100:.2f}% | "
                f"Top-5: {current_top5*100:.2f}%"
            )

    top1_acc = correct1 / total
    top5_acc = correct5 / total

    return top1_acc, top5_acc, num_classes


# Run evaluation
test_top1, test_top5, num_classes = evaluate()

# Print results
print("\n" + "=" * 60)
print("BioCLIP Zero-Shot Test Results")
print("=" * 60)
print(f"Test Top-1:    {test_top1:.4f} ({test_top1*100:.2f}%)")
print(f"Test Top-5:    {test_top5:.4f} ({test_top5*100:.2f}%)")
print("=" * 60)

# Save results to file
result_file = os.path.join(RESULTS_DIR, "bioclip_zeroshot_class_test.txt")

with open(result_file, "w") as f:
    f.write("BioCLIP Zero-Shot Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: BioCLIP (hf-hub:imageomics/bioclip)\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"Test Type: Zero-shot (no fine-tuning)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Top-1:    {test_top1:.4f} ({test_top1*100:.2f}%)\n")
    f.write(f"Test Top-5:    {test_top5:.4f} ({test_top5*100:.2f}%)\n")
    f.write("=" * 60 + "\n")
    f.write("\nNote:\n")
    f.write("This is a zero-shot evaluation where the model was NOT fine-tuned\n")
    f.write("on the training set. Classification is performed by matching image\n")
    f.write("embeddings with text embeddings of species names.\n")

print(f"\nResults saved to: {result_file}")
