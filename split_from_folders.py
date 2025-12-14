import os
import random
import csv

# =========================
# 配置区
# =========================
DATA_ROOT = "full"      # 你的真实数据目录
OUT_DIR = "splits"      # 输出 CSV
SEED = 42
MAX_PER_CLASS = None    # e.g. 100；None = 不限制

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 划分规则
# =========================
def split_counts(n):
    if n >= 30:
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val
    elif n >= 15:
        n_train, n_val, n_test = n - 4, 2, 2
    elif n >= 8:
        n_train, n_val, n_test = n - 2, 1, 1
    elif n >= 5:
        n_train, n_val, n_test = n - 1, 1, 0
    else:
        n_train, n_val, n_test = n, 0, 0
    return n_train, n_val, n_test

# =========================
# 主逻辑
# =========================
splits = {"train": [], "val": [], "test": []}

species_ids = sorted(os.listdir(DATA_ROOT))

for species_id in species_ids:
    species_dir = os.path.join(DATA_ROOT, species_id)
    if not os.path.isdir(species_dir):
        continue

    images = [
        f for f in os.listdir(species_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        continue

    random.shuffle(images)

    if MAX_PER_CLASS is not None:
        images = images[:MAX_PER_CLASS]

    n = len(images)
    n_train, n_val, n_test = split_counts(n)

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:n_train + n_val + n_test],
    }

    print(
        f"[{species_id}] total={n:4d} | "
        f"train={len(split_map['train']):3d} "
        f"val={len(split_map['val']):3d} "
        f"test={len(split_map['test']):3d}"
    )

    for split, imgs in split_map.items():
        for img in imgs:
            img_path = os.path.join(DATA_ROOT, species_id, img)
            splits[split].append((img_path, species_id))

# =========================
# 写 CSV
# =========================
for split, rows in splits.items():
    out_csv = os.path.join(OUT_DIR, f"{split}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species_id"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} samples to {out_csv}")

print("🎉 数据集划分完成")
