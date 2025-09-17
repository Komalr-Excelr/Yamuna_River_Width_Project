import os
import numpy as np
from PIL import Image
from skimage import morphology, exposure

def simple_cloud_removal(img):
    # Convert to grayscale
    gray = img.convert('L')
    arr = np.array(gray)
    # Normalize
    arr = exposure.rescale_intensity(arr, out_range=(0, 255))
    # Threshold: clouds are very bright
    cloud_mask = arr > 200
    # Morphological closing to fill small holes
    cloud_mask = morphology.closing(cloud_mask, morphology.disk(5))
    # Set cloud pixels to median of non-cloud
    non_cloud = arr[~cloud_mask]
    median_val = np.median(non_cloud) if non_cloud.size > 0 else 127
    arr[cloud_mask] = median_val
    return Image.fromarray(arr).convert('RGB')

def process_all_tif(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith('.tif'):
            img_path = os.path.join(src_dir, fname)
            img = Image.open(img_path)
            cleaned = simple_cloud_removal(img)
            cleaned.save(os.path.join(dst_dir, fname))
            print(f"Processed {fname}")

if __name__ == "__main__":
    src = "data/tif_images"
    dst = "data/clean_images"
    process_all_tif(src, dst)
