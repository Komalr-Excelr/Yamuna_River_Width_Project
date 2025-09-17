import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

CLEAN_IMG_DIR = 'data/clean_images'
MASK_DIR = 'data/masks'
BASE_OUT = 'data/dataset'
SPLITS = ['train', 'val', 'test']
SPLIT_RATIOS = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

# Get all image filenames
all_files = [f for f in os.listdir(CLEAN_IMG_DIR) if f.endswith('.tif')]
random.shuffle(all_files)
n_total = len(all_files)

n_train = int(n_total * SPLIT_RATIOS[0])
n_val = int(n_total * SPLIT_RATIOS[1])
n_test = n_total - n_train - n_val

splits = {
    'train': all_files[:n_train],
    'val': all_files[n_train:n_train+n_val],
    'test': all_files[n_train+n_val:]
}

for split in SPLITS:
    img_out = os.path.join(BASE_OUT, split, 'images')
    mask_out = os.path.join(BASE_OUT, split, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    for fname in splits[split]:
        shutil.copy(os.path.join(CLEAN_IMG_DIR, fname), os.path.join(img_out, fname))
        shutil.copy(os.path.join(MASK_DIR, fname), os.path.join(mask_out, fname))
    print(f"Copied {len(splits[split])} images/masks to {split}")
