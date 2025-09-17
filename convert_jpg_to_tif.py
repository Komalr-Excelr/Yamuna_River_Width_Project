import os
from PIL import Image

def convert_jpg_to_tif(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith('.jpg'):
            img_path = os.path.join(src_dir, fname)
            img = Image.open(img_path)
            base = os.path.splitext(fname)[0]
            tif_path = os.path.join(dst_dir, base + '.tif')
            img.save(tif_path, format='TIFF')
            print(f"Converted {fname} to {base + '.tif'}")

if __name__ == "__main__":
    src = "data/raw_images"
    dst = "data/tif_images"
    convert_jpg_to_tif(src, dst)
