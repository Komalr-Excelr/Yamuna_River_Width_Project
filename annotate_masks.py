import os
import numpy as np
from PIL import Image

def create_mask(image):
    # Convert to numpy array
    arr = np.array(image)
    # Simple color thresholding (customize as needed)
    # Water: dark blue/black (low R,G,B)
    water = (arr[:,:,0] < 60) & (arr[:,:,1] < 80) & (arr[:,:,2] < 100)
    # Land: bright (high R,G,B)
    land = (arr[:,:,0] > 120) & (arr[:,:,1] > 100) & (arr[:,:,2] > 80)
    # Riverbank: in-between
    riverbank = ~(water | land)
    # 0: background, 1: water, 2: riverbank, 3: land
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    mask[water] = 1
    mask[riverbank] = 2
    mask[land] = 3
    return mask

def process_all_clean_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith('.tif'):
            img_path = os.path.join(src_dir, fname)
            img = Image.open(img_path).convert('RGB')
            mask = create_mask(img)
            mask_img = Image.fromarray(mask)
            mask_img.save(os.path.join(dst_dir, fname))
            print(f"Mask created for {fname}")

if __name__ == "__main__":
    src = "data/clean_images"
    dst = "data/masks"
    process_all_clean_images(src, dst)
