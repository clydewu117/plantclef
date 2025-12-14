import os
import re
import pandas as pd
import torch
import open_clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =====================================================
# Config (改这里就行)
# =====================================================
DATA_ROOT = "/fs/scratch/PAS2099/plantclef"  # 用于拼接 image_path（如果 image_path 不是绝对路径）
TRAIN_SPLIT_CSV = (
    "/fs/scratch/PAS2099/plantclef/splits/train.csv"  # 逗号分隔：image_path,species_id
)
VAL_SPLIT_CSV = "/fs/scratch/PAS2099/plantclef/splits/val.csv"  # 同上

# ！！把这个路径改成你真实的 metadata 路径（分号分隔，带 species/genus/family 的那个）
META_CSV = "/fs/scratch/PAS2099/plantclef/full_metadata.csv"

BATCH_SIZE = 256
NUM_WORKERS = 8
TOPK = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Utils: robust CSV readers
# =====================================================
def read_split_csv(path: str) -> pd.DataFrame:
    """
    split CSV: comma-separated with header: image_path,species_id
    """
    df = pd.read_csv(path)
    # normalize columns
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
    """
    metadata CSV: semicolon-separated, quoted headers like "species_id";"species";...
    """
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
    # species / genus / family may exist; we'll use species primarily
    return df


def normalize_species_name(s: str) -> str:
    """
    Light cleanup for prompts:
    - collapse spaces
    - optionally remove trailing author token like 'L.'
    """
    s = re.sub(r"\s+", " ", str(s)).strip()
    parts = s.split(" ")
    if len(parts) >= 3 and re.fullmatch(r"[A-Za-z]\.", parts[-1]):
        s = " ".join(parts[:-1])
    return s


# =====================================================
# Build label_map from TRAIN split (this must match your dataloader logic)
# =====================================================
def build_label_map_from_train_split(train_split_df: pd.DataFrame):
    uniq = sorted(train_split_df["species_id"].unique())
    return {sid: i for i, sid in enumerate(uniq)}


# =====================================================
# Build species_id -> species text mapping from metadata
# =====================================================
def build_sid_to_species(meta_df: pd.DataFrame):
    # pick best available name column
    # You have: species, genus, family. We'll prefer species.
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

    # one name per species_id
    sid_to_name = (
        meta_df.dropna(subset=[name_col])
        .drop_duplicates(subset=["species_id"])
        .set_index("species_id")[name_col]
        .astype(str)
        .to_dict()
    )
    # normalize
    sid_to_name = {
        sid: normalize_species_name(name) for sid, name in sid_to_name.items()
    }
    return sid_to_name, name_col


# =====================================================
# Minimal Dataset: uses split CSV + label_map; loads image from DATA_ROOT/image_path
# =====================================================
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

        # resolve path
        img_path = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.join(self.data_root, rel_path)
        )
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)

        label = self.label_map[sid]
        return img, label


# =====================================================
# Build text features with templates
# =====================================================
@torch.no_grad()
def build_text_features(model, tokenizer, classnames, templates):
    feats = []
    for cname in tqdm(classnames, desc="Encoding class text"):
        prompts = [t.format(cname) for t in templates]
        tok = tokenizer(prompts).to(DEVICE)
        tf = model.encode_text(tok)  # [T, D]
        tf = tf / tf.norm(dim=-1, keepdim=True)
        tf = tf.mean(dim=0)
        tf = tf / tf.norm()
        feats.append(tf)
    return torch.stack(feats, dim=0)  # [C, D]


# =====================================================
# Main
# =====================================================
def main():
    # 1) Load BioCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.to(DEVICE).eval()

    # 2) Read CSVs
    train_split = read_split_csv(TRAIN_SPLIT_CSV)
    val_split = read_split_csv(VAL_SPLIT_CSV)
    meta = read_meta_csv(META_CSV)

    # 3) Build label_map from TRAIN split (matches your split-based training)
    label_map = build_label_map_from_train_split(train_split)
    idx_to_sid = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print("Num classes:", num_classes)

    # 4) Build species_id -> species name mapping from metadata
    sid_to_species, used_col = build_sid_to_species(meta)
    print(f"[INFO] Using metadata name column: {used_col}")

    # check coverage: how many sids in label_map are missing in metadata?
    missing = [sid for sid in label_map.keys() if sid not in sid_to_species]
    if len(missing) > 0:
        print(
            f"[WARN] {len(missing)}/{num_classes} species_id not found in metadata '{used_col}'."
        )
        print("       Example missing sids:", missing[:10])
        print(
            "       Falling back to using species_id string as name for missing ones (zero-shot quality may drop)."
        )

    # 5) Create classnames aligned with label indices
    classnames = []
    for i in range(num_classes):
        sid = idx_to_sid[i]
        name = sid_to_species.get(sid, sid)  # fallback to id if missing
        classnames.append(name)

    print("Example mapping:")
    for i in range(min(3, num_classes)):
        print(f"  idx={i} sid={idx_to_sid[i]} name='{classnames[i]}'")

    # 6) Build text features
    templates = [
        "a photo of {}",
        "a photo of the plant {}",
        "a close-up photo of {}",
    ]
    text_features = build_text_features(
        model, tokenizer, classnames, templates
    )  # [C, D]

    # 7) DataLoader for val split
    val_ds = SplitImageDataset(val_split, label_map, preprocess, DATA_ROOT)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # 8) Evaluate
    correct1 = 0
    correctk = 0
    total = 0

    with torch.no_grad():
        logit_scale = (
            model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
        )

        for imgs, labels in tqdm(val_loader, desc="Zero-shot eval"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            img_f = model.encode_image(imgs)  # [B, D]
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            logits = logit_scale * (img_f @ text_features.t())  # [B, C]

            pred1 = logits.argmax(dim=1)
            correct1 += (pred1 == labels).sum().item()

            topk = logits.topk(TOPK, dim=1).indices
            correctk += topk.eq(labels[:, None]).any(dim=1).sum().item()

            total += labels.size(0)

    top1 = correct1 / total
    topk_acc = correctk / total
    print(
        f"\nBioCLIP Zero-shot | Top1: {top1:.4f} | Top{TOPK}: {topk_acc:.4f} | N={total}"
    )


if __name__ == "__main__":
    main()
