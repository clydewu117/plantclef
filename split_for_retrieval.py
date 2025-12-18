import os
import random
import csv

DATA_ROOT = "/fs/scratch/PAS2099/plantclef/full"
OUT_DIR = "/fs/scratch/PAS2099/plantclef/re_splits"
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

def split_counts(n):
    """Split samples into query and gallery for retrieval task"""
    if n >= 30:
        n_train = int(0.8 * n)
        n_query = int(0.1 * n)
        n_gallery = n - n_train - n_query
    elif n >= 15:
        n_train, n_query, n_gallery = n - 4, 2, 2
    elif n >= 8:
        n_train, n_query, n_gallery = n - 2, 1, 1
    elif n >= 5:
        n_train, n_query, n_gallery = n - 1, 1, 0
    else:
        n_train, n_query, n_gallery = n, 0, 0
    return n_train, n_query, n_gallery

# Get all species directories
species_ids = sorted(os.listdir(DATA_ROOT))
species_with_images = []

print("Scanning species directories...")
for species_id in species_ids:
    species_dir = os.path.join(DATA_ROOT, species_id)
    if not os.path.isdir(species_dir):
        continue

    images = [
        f for f in os.listdir(species_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) > 0:
        species_with_images.append((species_id, len(images)))

print(f"Found {len(species_with_images)} species with images")

# Shuffle and split species into train and test sets
random.shuffle(species_with_images)
mid_point = len(species_with_images) // 2

train_species = species_with_images[:mid_point]
test_species = species_with_images[mid_point:]

print(f"\nSplit statistics:")
print(f"  Train species: {len(train_species)}")
print(f"  Test species (for retrieval): {len(test_species)}")

# Prepare split containers
splits = {
    "train": [],           # Training set from train species
    "query": [],           # Query set from test species
    "gallery": [],         # Gallery set from test species
    "train_val": [],       # Validation set from train species (optional)
}

print("\n" + "="*80)
print("Processing TRAIN species (for model training):")
print("="*80)

for species_id, img_count in train_species:
    species_dir = os.path.join(DATA_ROOT, species_id)

    images = [
        f for f in os.listdir(species_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    # For train species: use most for training, keep small portion for validation
    n = len(images)
    if n >= 10:
        n_train_split = int(0.9 * n)
        n_val_split = n - n_train_split
    else:
        n_train_split = n
        n_val_split = 0

    train_imgs = images[:n_train_split]
    val_imgs = images[n_train_split:n_train_split + n_val_split]

    print(
        f"[TRAIN] {species_id:20s} | total={n:4d} | "
        f"train={len(train_imgs):4d} val={len(val_imgs):4d}"
    )

    for img in train_imgs:
        img_path = os.path.join(DATA_ROOT, species_id, img)
        splits["train"].append((img_path, species_id))

    for img in val_imgs:
        img_path = os.path.join(DATA_ROOT, species_id, img)
        splits["train_val"].append((img_path, species_id))

print("\n" + "="*80)
print("Processing TEST species (for retrieval evaluation):")
print("="*80)

for species_id, img_count in test_species:
    species_dir = os.path.join(DATA_ROOT, species_id)

    images = [
        f for f in os.listdir(species_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    n = len(images)
    n_query, n_gallery = 0, 0

    # Split into query and gallery
    if n >= 10:
        n_query = max(1, int(0.2 * n))
        n_gallery = n - n_query
    elif n >= 5:
        n_query = 1
        n_gallery = n - 1
    elif n >= 2:
        n_query = 1
        n_gallery = n - 1
    else:
        # If only 1 image, use it as gallery only
        n_query = 0
        n_gallery = n

    query_imgs = images[:n_query]
    gallery_imgs = images[n_query:n_query + n_gallery]

    print(
        f"[TEST]  {species_id:20s} | total={n:4d} | "
        f"query={len(query_imgs):4d} gallery={len(gallery_imgs):4d}"
    )

    for img in query_imgs:
        img_path = os.path.join(DATA_ROOT, species_id, img)
        splits["query"].append((img_path, species_id))

    for img in gallery_imgs:
        img_path = os.path.join(DATA_ROOT, species_id, img)
        splits["gallery"].append((img_path, species_id))

# Write all splits to CSV files
print("\n" + "="*80)
print("Writing CSV files...")
print("="*80)

for split_name, rows in splits.items():
    if len(rows) == 0:
        print(f"⚠️  Skipping {split_name}.csv (empty)")
        continue

    out_csv = os.path.join(OUT_DIR, f"{split_name}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species_id"])
        writer.writerows(rows)

    print(f"✅ Saved {len(rows):6d} samples to {split_name}.csv")

# Save species split information
train_species_file = os.path.join(OUT_DIR, "train_species.txt")
test_species_file = os.path.join(OUT_DIR, "test_species.txt")

with open(train_species_file, "w") as f:
    for species_id, _ in train_species:
        f.write(f"{species_id}\n")

with open(test_species_file, "w") as f:
    for species_id, _ in test_species:
        f.write(f"{species_id}\n")

print(f"\n✅ Saved train species list to train_species.txt")
print(f"✅ Saved test species list to test_species.txt")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print(f"Train species:     {len(train_species)}")
print(f"Test species:      {len(test_species)}")
print(f"Train samples:     {len(splits['train'])}")
print(f"Train-val samples: {len(splits['train_val'])}")
print(f"Query samples:     {len(splits['query'])}")
print(f"Gallery samples:   {len(splits['gallery'])}")
print("\n🎉 Retrieval dataset split completed!")
